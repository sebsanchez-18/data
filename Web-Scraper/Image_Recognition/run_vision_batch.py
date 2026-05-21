from google.cloud import vision_v1
from google.cloud import storage

BUCKET_NAME = "vision_map"
INPUT_PREFIX = "input/"
OUTPUT_PREFIX = "vision_output/"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def list_images(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    uris = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.lower().endswith(IMAGE_EXTENSIONS):
            uris.append(f"gs://{bucket_name}/{blob.name}")

    return uris


def run_batch(image_uris):
    client = vision_v1.ImageAnnotatorClient()

    features = [
        vision_v1.Feature(type_=vision_v1.Feature.Type.LABEL_DETECTION),
        vision_v1.Feature(type_=vision_v1.Feature.Type.TEXT_DETECTION),
        vision_v1.Feature(type_=vision_v1.Feature.Type.OBJECT_LOCALIZATION),
    ]


    requests = []

    for uri in image_uris:
        image = vision_v1.Image(
            source=vision_v1.ImageSource(image_uri=uri)
        )

        request = vision_v1.AnnotateImageRequest(
            image=image,
            features=features
        )

        requests.append(request)

    output_config = vision_v1.OutputConfig(
        gcs_destination=vision_v1.GcsDestination(
            uri=f"gs://{BUCKET_NAME}/{OUTPUT_PREFIX}"
        ),
        batch_size=100,
    )

    operation = client.async_batch_annotate_images(
        requests=requests,
        output_config=output_config,
    )

    print(f"Submitted {len(requests)} images to Google Vision.")
    print("Waiting for results...")

    operation.result(timeout=3600)

    print("Done.")
    print(f"Results saved to gs://{BUCKET_NAME}/{OUTPUT_PREFIX}")


if __name__ == "__main__":
    image_uris = list_images(BUCKET_NAME, INPUT_PREFIX)
    print(f"Found {len(image_uris)} images.")

    if len(image_uris) == 0:
        raise RuntimeError("No images found. Check your bucket name and input folder.")

    run_batch(image_uris)