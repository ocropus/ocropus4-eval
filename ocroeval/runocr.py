import glob, os, re, subprocess, time, warnings
from itertools import islice

import numpy as np
import typer
import webdataset as wds
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

app = typer.Typer()


def enumerate_report(source, interval=5, growth=1.3):
    last = 0
    count = 0
    for item in source:
        if time.time() - last > interval:
            last = time.time()
            interval *= growth
            yield count, item, True
        else:
            yield count, item, False
        count += 1


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
    source: str,
    output: str = typer.Option(None, help="The output shard."),
    hocr: str = typer.Option("hocr.html", help="The input key for the hOCR data."),
    img: str = typer.Option("page.jpg", help="The input key for the page image."),
    margin: int = typer.Option(10, help="Extra margin for reframing."),
    maxcount: int = typer.Option(999999, help="Process at most this many images."),
):
    """Reframe images based on hOCR bounding boxes.

    This command uses the bounding boxes of the words and text lines
    in hOCR ground truth to reframe the images; that is, it erases everything
    outside a bounding box of all the text. This is useful for evaluating
    OCR on images that contain extraneous material outside the page frame.
    """
    ds = wds.WebDataset(source).decode("rgb")
    output = wds.TarWriter(output)
    for count, sample, report in enumerate_report(islice(ds, maxcount)):
        if hocr not in sample:
            warnings.warn(f"missing {hocr} in {sample['__key__']}, got {sample.keys()}")
            continue
        frame = compute_image_frame(sample[hocr])
        if frame is None:
            continue
        (x0, y0), (x1, y1) = frame
        if report:
            print(count, sample["__key__"], x0, y0, x1, y1)
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


def run_command(thecommand):
    process = subprocess.run(
        thecommand,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = process.stdout.decode("utf-8")
    stderr = process.stderr.decode("utf-8")
    return stdout, stderr


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
        try:
            run_command(thecommand)
        except subprocess.CalledProcessError as e:
            print(e)
            print(e.stdout)
            print(e.stderr)
            raise
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
    for count, sample, report in enumerate_report(islice(ds, maxcount)):
        sample = runocr1(sample)
        if report:
            print(count, sample["__key__"], repr(sample[output_key])[-100:])
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


def main():
    app()


if __name__ == "__main__":
    app()
