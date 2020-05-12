import sys
import asyncio
from frozendict import frozendict
from argparse import ArgumentParser

# sys.path.append("./app")

from scenarios.scenario_zone_estimation import main_zone_estimation
from scenarios.scenario_face_detection import main_face_detection
from scenarios.scenario_zone_protection import main_zone_protection


CONTENT_TYPE = 'image/jpeg'
HEADERS = {'content-type': CONTENT_TYPE}
OPERATIONS = ["object_detection", "zone_estimation", "zone_protection"]
OPERATION_TO_IMPL = frozendict(
    object_detection=main_face_detection,
    zone_estimation=main_zone_estimation,
    zone_protection=main_zone_protection
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-a", "--address", default="0.0.0.0", type=str)
    parser.add_argument("-p", "--port", default=1000, type=int)
    parser.add_argument("-o", "--operation", choices=OPERATIONS, type=str)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    main_url = f"http://{args.address}:{args.port}"
    operation = OPERATION_TO_IMPL[args.operation]
    if asyncio.iscoroutinefunction(operation):
        asyncio.run(operation(main_url))
    else:
        operation(main_url)


if __name__ == "__main__":
    main()
