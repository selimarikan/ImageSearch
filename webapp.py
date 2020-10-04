import base64
import io
from io import BytesIO
import json
import os

from sklearn.neighbors import NearestNeighbors
import pandas as pd
from PIL import Image
import flask
from flask.globals import request
from joblib import load as load_mdl

from extractor import extract_single_image_PIL, get_trafo, get_model


class ImageSearchApp(object):
    """
    Main application for the image search
    """

    def __init__(self):
        """
        docstring
        """
        self.app = flask.Flask(__name__)
        self.config: dict = {}
        self.nearest_model: NearestNeighbors
        self.features_meta: pd.DataFrame = pd.DataFrame()
        self.torch_model: list = []
        self.torch_trafo: list = []

        # Connect endpoints
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule(
            "/file-upload", "file-upload", self.image_upload, methods=["POST"]
        )

    def read_config(self) -> None:
        """
        Read config from current directory config.json
        """
        filename: str = "config.json"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        with open(filename) as f:
            config = json.load(f)

        self.config = config

    def read_features_meta(self):
        """
        Read the features and metadata from disk
        """
        self.features_meta = pd.read_parquet(self.config["model"]["features_file"])

    def load_model(self):
        """
        Load the nearest neighbours model
        """
        self.nearest_model = load_mdl(self.config["nearest"]["model_path"])

    def get_b64_img(self, path: str):
        """
        Returns a base64 encoded image for the given path
        """
        buffer = BytesIO()
        Image.open(path).save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    def run(self):
        # Set model and trafos
        self.torch_model = get_model()

        self.torch_trafo = get_trafo(
            input_size=(
                self.config["model"]["input_size"],
                self.config["model"]["input_size"],
            ),
            dataset_mean=self.config["model"]["dataset_mean"],
            dataset_std=self.config["model"]["dataset_std"],
        )

        host = self.config["webapp"]["host"]
        port = self.config["webapp"]["port"]
        debug = self.config["webapp"]["debug"]
        self.app.run(host, port, debug=debug)

    # ENDPOINT
    def index(self):
        return flask.render_template("index.html", datacontext=self.config)

    # ENDPOINT
    def image_upload(self):
        if "file" not in flask.request.files:
            print(f"Error: No image file provided")
            return flask.Response(status=415)

        # Read the image from the frontend request
        image_file = flask.request.files["file"].read()

        # Convert the raw image file to PIL Image
        image = Image.open(io.BytesIO(image_file)).convert("RGB")

        # Get CNN features of the image
        features = extract_single_image_PIL(image, self.torch_trafo, self.torch_model)

        # Run the nearest neighbours model
        distances, indices = self.nearest_model.kneighbors(
            features, self.config["nearest"]["k_nearest"]
        )
        dists = list(distances[0])
        paths = list(self.features_meta.index[indices][0])

        print(dists)
        print(paths)

        # Read images in Python side due to JS browser security issues
        imgs = [self.get_b64_img(p) for p in paths]

        return flask.jsonify(
            {
                "imgs": imgs,
                "distances": dists,
            }
        )


if __name__ == "__main__":
    imsearch_app = ImageSearchApp()
    imsearch_app.read_config()
    imsearch_app.read_features_meta()
    imsearch_app.load_model()
    imsearch_app.run()