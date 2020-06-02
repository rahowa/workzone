import jsonpickle
from flask import Response


def load_classes(path: str) -> Response:
    with open(path) as cls_file:
        classes = cls_file.read().splitlines()
        response = {"total": len(classes),
                    "classes": classes}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")