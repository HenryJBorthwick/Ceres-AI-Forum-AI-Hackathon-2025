import pathlib
import requests


def download_ipc_csv(url: str, output_path: pathlib.Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a large CSV file in streamed chunks.

    Parameters
    ----------
    url : str
        Source URL of the CSV file.
    output_path : pathlib.Path
        Local filesystem path where the file will be written.
    chunk_size : int, optional
        Size (in bytes) of streamed chunks (default 1 MB).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} → {output_path} …")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

    size_mb = output_path.stat().st_size / 1_048_576  # bytes → MiB
    print(f"Download complete. File size: {size_mb:.2f} MiB")


if __name__ == "__main__":
    DATA_URL = (
        "https://s3.us-east-1.amazonaws.com/hdx-production-filestore/resources/"
        "2c471c60-b94b-4f9d-b4a1-9a4373e213d1/ipc_global_area_long.csv?"
        "AWSAccessKeyId=AKIAXYC32WNATGSC6EFO&Signature=mmVcYfDj6GSV1DhC7FuQmFFx4AA%3D&Expires=1755123951"
    )

    project_root = pathlib.Path(__file__).resolve().parent.parent
    output_csv = project_root / "data" / "ipc_global_area_long.csv"

    download_ipc_csv(DATA_URL, output_csv) 