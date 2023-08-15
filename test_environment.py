import sys

required_major = 3
required_minor = 8


def main():
    system_major, system_minor = sys.version_info.major, sys.version_info.minor
    if (system_major, system_minor) != (required_major, required_minor):
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                f"{required_major}.{required_minor}", sys.version
            )
        )
    else:
        print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()
