import glob, os, re, sqlite3, warnings
from itertools import islice

import Levenshtein as lev
import numpy as np
import regex
import scipy
import typer
import webdataset as wds
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from webdataset.pytorch import IterableDataset

app = typer.Typer()


def compute_image_frame(hocr):
    """Given an hOCR input, compute the image frame as the union of all bounding boxes."""

    soup = BeautifulSoup(hocr, "html.parser")
    boxes = []
    for cls in ["ocr_line", "ocrx_word"]:
        for span in soup.find_all("span", {"class": cls}):
            title = span["title"]
            m = re.match(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
            if m:
                boxes.append([int(x) for x in m.groups()])
    if not boxes:
        return None
    boxes = np.array(boxes)
    return np.array(
        [
            [boxes[:, 0].min(), boxes[:, 1].min()],
            [boxes[:, 2].max(), boxes[:, 3].max()],
        ]
    )


@app.command()
def reframe(
    source,
    output: str = None,
    hocr: str = "hocr.html",
    img: str = "page.jpg",
    margin: int = 10,
    maxcount: int = 999999,
):
    ds = wds.WebDataset(source).decode("rgb")
    output = wds.TarWriter(output)
    for sample in islice(ds, maxcount):
        if hocr not in sample:
            warnings.warn(f"missing {hocr} in {sample['__key__']}, got {sample.keys()}")
            continue
        frame = compute_image_frame(sample[hocr])
        if frame is None:
            continue
        (x0, y0), (x1, y1) = frame
        print(sample["__key__"], x0, y0, x1, y1)
        h, w = sample[img].shape[:2]
        result_image = np.zeros_like(sample[img])
        result_image[:, :, ...] = np.amax(sample[img])
        y0, x0, y1, x1 = (
            int(max(y0 - margin, 0)),
            int(max(x0 - margin, 0)),
            int(min(y1 + margin, h)),
            int(min(x1 + margin, w)),
        )
        result_image[y0:y1, x0:x1] = sample[img][y0:y1, x0:x1]
        sample[img] = result_image
        output.write(sample)
    output.close()


def runocr1(
    sample,
    img_key="page.jpg",
    output_key="tess.html",
    cmd: str = "tesseract page.jpg output hocr",
):
    """Run OCR on a single image."""
    import tempfile

    with tempfile.TemporaryDirectory() as dirname:
        image = sample[img_key]
        with open(os.path.join(dirname, "page.jpg"), "wb") as stream:
            stream.write(image)
        thecommand = f"cd {dirname} && " + cmd.format(
            input="page.jpg", output_base="out"
        )
        print("#", thecommand)
        assert os.system(thecommand) == 0, thecommand
        outputs = glob.glob(os.path.join(dirname, "output.*"))
        assert len(outputs) == 1, outputs
        with open(os.path.join(dirname, outputs[0])) as stream:
            sample[output_key] = stream.read()
    return sample


@app.command()
def runocr(
    source,
    output: str = None,
    img: str = "page.jpg",
    output_key: str = "tess.html",
    cmd: str = "tesseract page.jpg output hocr",
    maxcount: int = 999999,
):
    """Run OCR over a webdataset."""
    ds = wds.WebDataset(source)
    output = wds.TarWriter(output)
    for sample in islice(ds, maxcount):
        sample = runocr1(sample)
        print(sample["__key__"], repr(sample[output_key])[-100:])
        output.write(sample)
    output.close()


@app.command()
def parocr(
    source,
    output: str = None,
    img_key: str = "page.jpg",
    output_key: str = "tess.html",
    cmd: str = "tess page.jpg output.html hocr",
    njobs=4,
    maxcount: int = 999999,
):
    """Run OCR on a webdataset in parallel using joblib."""
    ds = wds.WebDataset(source).decode("rgb")
    output = wds.TarWriter(output)
    f = partial(runocr1, cmd=cmd)
    results = Parallel(n_jobs=njobs, batch_size=njobs)(
        delayed(f)(sample) for sample in islice(ds, maxcount)
    )
    for result in results:
        output.write({"hocr": result})


def strip_hocr(html_doc):
    """Strip all HTML tags from an input document.

    This function strips all HTML tags from an input document. It also
    normalizes whitespace by removing all newlines and replacing all
    whitespace sequences with a single space within each `ocr_line`
    element, making the output more readable.
    """
    soup = BeautifulSoup(html_doc, "html.parser")
    # remove all spans with class 'ocrx_word'
    for span in soup.find_all("span", {"class": "ocrx_word"}):
        span.unwrap()
    for span in soup.find_all("span", {"class": "ocr_line"}):
        text = span.text
        text = re.sub(r"[\n ]+", " ", text)
        span.string = text
    text = soup.get_text()
    text = re.sub(r"\n\n+", "\n", text)
    return text


def normalize(text):
    """Normalize text for OCR evaluation.

    This function normalizes text for OCR evaluation. It removes all
    non-alphanumeric characters and converts all whitespace to a single
    space. It also strips leading and trailing whitespace.
    """
    text = regex.sub(r"[^\p{L}\p{N}]+", " ", text).strip()
    return text


def compute_distances(gtw, seqw, r, start=None, end=None):
    """Compute edit distances between all substrings of length `r` in `gtw` and `seqw`.

    This computes edit distances between all substrings of length `r` in `gtw` and
    all substrings of length `r` in `seqw`. The `start` and `end` arguments specify
    the range of substrings to consider in `gtw`. If `start` is `None`, then it is
    set to `0`. If `end` is `None`, then it is set to `len(gtw) - r`. The return value
    is a matrix of edit distances, with one row for each substring of `gtw` and one
    column for each substring of `seqw`.
    """
    if start is None:
        start = 0
    if end is None:
        end = len(gtw) - r

    distances = np.zeros((end - start, len(seqw) - r))

    for i in range(start, end):
        g = gtw[i : i + r]
        for j in range(len(seqw) - r):
            s = seqw[j : j + r]
            d1 = lev.distance(g, s)
            distances[i - start, j] = d1 * 1.0 / r

    return distances


def compute_distances_parallel(gtw, seqw, r=20, nchunks=None):
    """Compute edit distances between all substrings of length `r` in `gtw` and `seqw`.

    This uses joblib for parallelization. The `nchunks` argument specifies the
    number of chunks to use for parallelization. If it is `None`, then the number
    of chunks is set to `os.cpu_count() - 4`. If it is `1`, then no parallelization
    is used.
    """
    assert (
        len(gtw) >= r and len(seqw) >= r
    ), f"inputs too short: gtw={len(gtw)} seqw={len(seqw)} r={r}"
    if nchunks is None:
        nchunks = max(2, os.cpu_count() - 4)
    elif nchunks <= 1:
        return compute_distances(gtw, seqw, r)
    chunk_size = len(gtw) // nchunks
    results = Parallel(n_jobs=nchunks)(
        delayed(compute_distances)(
            gtw, seqw, r, i * chunk_size, min((i + 1) * chunk_size, len(gtw))
        )
        for i in range(nchunks)
    )

    return np.concatenate(results, axis=0)


def distance_from_diagonal(m, n):
    """Compute distance from diagonal in edit distance matrix."""
    rows, cols = np.indices((m, n))
    return np.abs(rows - cols)


def bias_towards_diagonal(image, b=2.0, scale=0.1):
    """Bias an edit distance matrix towards the diagonal."""
    m, n = image.shape
    return image - scale * np.exp(-distance_from_diagonal(m, n))


def ocr_eval(gt, ocr, r=20, threshold=10, intermediate=False):
    """Evaluate OCR results relative to ground truth.

    This function evaluates OCR results relative to ground truth. The
    `gt` and `ocr` arguments are strings containing the ground truth
    and the OCR results, respectively. The `r` argument specifies the
    locality of matching; it should be around 20 or larger.

    This works by computing the edit distance between all substrings
    of length `r` in the ground truth and all substrings of length `r`
    in the OCR results.

    This results in a matrix of edit distances. A true edit distance would
    find a constrained best path through this matrix, but that is
    computationally intractable. We approximate this by simply finding
    the minimum edit distance for each row and each column.

    Furthermore, we compute layout errors by finding breaks in the locally
    best matching locations.

    Overall, this returns three values: the average edit distance between
    the OCR results and the ground truth, the average edit distance between
    the ground truth and the OCR results, and the number of layout errors.

    Optionaly, if `intermediate` is true, then this returns a dictionary
    containing all the intermediate results, useful for debugging and
    visualization.
    """
    gt_normalized = normalize(gt)
    ocr_normalized = normalize(ocr)
    if len(gt_normalized) <= 2 * r or len(ocr_normalized) <= 2 * r:
        err = lev.distance(gt_normalized, ocr_normalized)
        if intermediate:
            return err, err, 0, locals()
        else:
            return err, err, 0
    distances = compute_distances_parallel(gt_normalized, ocr_normalized, r)
    err = np.mean(distances.min(axis=0))
    reverse_err = np.mean(distances.min(axis=1))
    biased_distances = bias_towards_diagonal(distances)
    locs = scipy.ndimage.median_filter(biased_distances.argmin(0), r)
    layout_errs = np.sum(np.abs(np.diff(locs)) > threshold)
    if intermediate:
        return err, reverse_err, layout_errs, locals()
    else:
        return err, reverse_err, layout_errs


@app.command()
def loadgt(urls: str, output: str="", key: str = "hocr.html", strip: bool = False, append:bool=False):
    """Extract ground truth from webdataset and store in sqlite database.

    This command extracts the ground truth from a webdataset and stores it
    in a sqlite database. The database is created if it does not exist.
    The database contains a single table named `ground_truth` with two
    columns: `key` and `gt`. The `key` column contains the webdataset key
    and the `gt` column contains the ground truth text. Ground truth text
    is extracted from the `key` field of the webdataset using the `key`
    argument. If `strip` is true, then the ground truth text is stripped
    of all HTML tags (useful if the ground truth is in hOCR format).
    """
    gt_ext = key
    if not append:
        assert not os.path.exists(output), f"{output} already exists"
    db = sqlite3.connect(output)
    db.execute(
        "CREATE TABLE IF NOT EXISTS ground_truth (key TEXT PRIMARY KEY, gt TEXT)"
    )
    count = 0
    ds = wds.WebDataset(urls, resampled=False).decode()
    for sample in ds:
        key = sample["__key__"]
        assert gt_ext in sample, f"missing {gt_ext} in {key}, have: {sample.keys()}"
        text = sample[gt_ext]
        if strip:
            text = strip_hocr(text)
        db.execute("INSERT OR REPLACE INTO ground_truth VALUES (?, ?)", (key, text))
        if count % 1000 == 0:
            print(count, key)
        count += 1
    db.commit()
    db.close()


class AugmentedDataset(IterableDataset):
    """Augment a webdataset with ground truth from a sqlite database."""

    def __init__(self, dataset, augment, key="hocr.html"):
        self.dataset = dataset
        self.augment = sqlite3.connect(augment)
        self.key = key

    def __iter__(self):
        for sample in self.dataset:
            key = sample["__key__"]
            text = self.augment.execute("SELECT gt FROM ground_truth WHERE key=?", (key,)).fetchone()[0]
            sample[self.key] = text
            yield sample

class ZippedDataset(IterableDataset):
    """Zip two webdatasets together."""

    def __init__(self, dataset, dataset2, keys=None):
        self.dataset = dataset
        self.dataset2 = dataset2
        self.keys = keys

    def __iter__(self):
        for sample, sample2 in zip(self.dataset, self.dataset2):
            assert sample["__key__"] == sample2["__key__"], f"incompatible shards: {sample['__key__']} != {sample2['__key__']}"  
            if self.keys:
                sample.update({k: sample2[k] for k in self.keys})
            else:
                sample.update(sample2)
            yield sample


@app.command()
def texteval(
    urls: str,
    gtdb: str = "",
    gtshards: str = "",
    gtkey: str = "hocr.html",
    ocrkey: str = "tess.html",
    output: str = "text_eval.db",
    strip: bool = True,
    maxcount: int = 999999999,
    r: int = 20,
    threshold: int = 20,
    record: bool = False,
    append: bool = False,
    keeptext: bool = True,
):
    """Evaluate OCR results from webdataset relative to a ground truth database.

    This command evaluates OCR results from a webdataset relative to ground truth.
    Ground truth can be specified either as a sqlite database or as separate webdataset
    shard, or be contained in the same shard as the OCR results.

    Output is in the form of an sqlite3 file

    The output contains the following columns:

    - `key`: the webdataset key
    - `gt`: the ground truth text
    - `gtsize`: the length of the ground truth text
    - `result`: the OCR result text
    - `resultsize`: the length of the OCR result text
    - `err`: the average edit distance between the OCR result and the ground truth
    - `rerr`: the average edit distance between the ground truth and the OCR result
    - `layout_err`: the number of layout errors (text blocks moved by more than `threshold` characters)

    Using `max(err, rerr)` or `mean(err, rerr)` as a metric for OCR quality is a reasonable choice.
    """
    if not append:
        assert not os.path.exists(output), f"{output} already exists"
    odb = sqlite3.connect(output)
    odb.execute(
        "CREATE TABLE IF NOT EXISTS ocr_eval " +
        "(key TEXT PRIMARY KEY, gt TEXT, gtsize INTEGER, result TEXT, resultsize INTEGER, err REAL, rerr REAL, layout_err INTEGER)"
    )
    ds = wds.WebDataset(urls).decode()
    if gtdb:
        ds = AugmentedDataset(ds, sqlite3.connect(gtdb), key=gtkey)
    elif gtshards:
        ds = ZippedDataset(ds, wds.WebDataset(gtshards).decode(), keys=[gtkey])
    for i, sample in enumerate(islice(ds, maxcount)):
        key = sample["__key__"]
        gt = sample[gtkey]
        ocr = sample[ocrkey]
        if strip:
            gt = strip_hocr(gt)
            ocr = strip_hocr(ocr)
        errs, rerrs, layout = ocr_eval(gt, ocr, r=r, threshold=threshold)
        print(i, key, errs, rerrs, layout)
        odb.execute(
            """
            INSERT OR REPLACE INTO ocr_eval 
            (key, gt, gtsize, result, resultsize, err, rerr, layout_err) 
            VALUES (:key, :gt, :gtsize, :result, :resultsize, :err, :rerr, :layout_err)
            """,
            {
                "key": key,
                "gt": gt if keeptext else "",
                "gtsize": len(gt),
                "result": ocr if keeptext else "",
                "resultsize": len(ocr),
                "err": errs,
                "rerr": rerrs,
                "layout_err": int(layout),
            },
        )
        odb.commit()
    odb.close()


def extract_bounding_boxes_and_text(hocr, element="ocrx_word"):
    """Extract bounding boxes and text from hOCR document.

    This function extracts bounding boxes and text from an hOCR document.
    The `element` argument specifies the type of element to extract.
    """
    soup = BeautifulSoup(hocr, "html.parser")
    boxes = []
    texts = []
    for span in soup.find_all("span", {"class": element}):
        title = span["title"]
        m = re.match(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
        if m:
            boxes.append([int(x) for x in m.groups()])
            texts.append(span.text)
    return np.array(boxes), texts

def compute_overlap(gt_box, ocr_box):
    # gt_box and ocr_box are (x0,y0,x1,y1) arrays
    # compute the overlap of the two boxes as area of intersection over minimum area of the two boxes
    # return 0 if there is no overlap
    # return 1 if the boxes are identical
    gt_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    ocr_area = (ocr_box[2]-ocr_box[0])*(ocr_box[3]-ocr_box[1])
    intersection_area = max(0, min(gt_box[2], ocr_box[2]) - max(gt_box[0], ocr_box[0])) * max(0, min(gt_box[3], ocr_box[3]) - max(gt_box[1], ocr_box[1]))
    if intersection_area == 0:
        return 0
    else:
        return intersection_area / min(gt_area, ocr_area)

def compute_overlap_matrix(gt_boxes, ocr_boxes):
    overlaps = np.zeros((len(gt_boxes), len(ocr_boxes)))
    for i,gt in enumerate(gt_boxes):
        for j,ocr in enumerate(ocr_boxes):
            overlaps[i,j] = compute_overlap(gt, ocr)
    return overlaps


def compute_match(gt_boxes, ocr_boxes, gt_text, ocr_text, verbose=False):
    overlaps = compute_overlap_matrix(gt_boxes, ocr_boxes)
    best_match_for_gt = np.argmax(overlaps, axis=1)
    missing_ocr = sum(overlaps.max(axis=0) == 0)
    extra_ocr = sum(overlaps.max(axis=1) == 0)
    errors, total = 0, 0
    for i, j in enumerate(best_match_for_gt):
        if overlaps[i, j] == 0:
            errors += len(gt_text[i])
            total += len(gt_text[i])
            continue
        gtt = normalize(gt_text[i])
        ocrt = normalize(ocr_text[j])
        error = lev.distance(gtt, ocrt)
        errors += error
        total += len(gtt)
        if error > 0 and verbose:
            print("GT: ", gtt)
            print("OCR:", ocrt)
            print()
    return errors, total, missing_ocr, extra_ocr


@app.command()
def bboxeval(
    urls: str,
    gtdb: str = "",
    gtshards: str = "",
    gtkey: str = "hocr.html",
    ocrkey: str = "tess.html",
    output: str = "bb_eval.db",
    maxcount: int = 999999999,
    element: str = "ocrx_word",
    append: bool = False,
    keeptext: bool = True,
    verbose: bool = False,
):
    if not append:
        assert not os.path.exists(output), f"{output} already exists"
    odb = sqlite3.connect(output)
    odb.execute(
        "CREATE TABLE IF NOT EXISTS bb_eval " +
        "(key TEXT PRIMARY KEY, gt TEXT, result TEXT, errors INT, total INT, missing_ocr INT, extra_ocr INT)"
    )
    ds = wds.WebDataset(urls).decode()
    if gtdb:
        ds = AugmentedDataset(ds, sqlite3.connect(gtdb), key=gtkey)
    elif gtshards:
        ds = ZippedDataset(ds, wds.WebDataset(gtshards).decode(), keys=[gtkey])
    for i, sample in enumerate(islice(ds, maxcount)):
        key = sample["__key__"]
        gt = sample[gtkey]
        ocr = sample[ocrkey]
        gt_boxes, gt_text = extract_bounding_boxes_and_text(gt, element=element)
        ocr_boxes, ocr_text = extract_bounding_boxes_and_text(ocr, element=element)
        if len(gt_boxes) == 0 or len(ocr_boxes) == 0:
            print(i, key, "no boxes")
            continue
        errors, total, missing_ocr, extra_ocr = compute_match(gt_boxes, ocr_boxes, gt_text, ocr_text, verbose=verbose)
        print(i, key, errors, total, missing_ocr, extra_ocr)
        odb.execute(
            """
            INSERT OR REPLACE INTO bb_eval 
            (key, gt, result, errors, total, missing_ocr, extra_ocr) 
            VALUES (:key, :gt, :result, :errors, :total, :missing_ocr, :extra_ocr)
            """,
            {
                "key": key,
                "gt": gt if keeptext else "",
                "result": ocr if keeptext else "",
                "errors": int(errors),
                "total": int(total),
                "missing_ocr": int(missing_ocr),
                "extra_ocr": int(extra_ocr),
            },
        )
        odb.commit()
    odb.close()



def main():
    app()


if __name__ == "__main__":
    app()
