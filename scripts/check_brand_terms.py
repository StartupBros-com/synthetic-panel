"""Keep git-grep hits with brand mentions outside the public repository URLs."""

import re
import sys

BRAND = b"startup" + b"bros"
PUBLIC_REPOSITORY = re.compile(
    rb"(?:^|(?<=[\s`\"'(<\[{]))(?:git\+)?(?:https://)?github\.com/"
    + re.escape(BRAND)
    + rb"(?:-com)?/synthetic-panel(?:\.git)?(?=$|[\s/@?#`\"'<>()\[\]{},;])",
    re.IGNORECASE,
)


def main():
    remaining = sys.stdin.buffer.read()
    while remaining:
        # git grep -nz emits path NUL line-number NUL content NEWLINE. NUL
        # separators keep colons and newlines in filenames out of the content.
        path, separator, remaining = remaining.partition(b"\0")
        if not separator:
            raise ValueError("missing git-grep path separator")
        number, separator, remaining = remaining.partition(b"\0")
        if not separator or not number.isdigit():
            raise ValueError("invalid git-grep line number")
        content, _, remaining = remaining.partition(b"\n")
        remainder = PUBLIC_REPOSITORY.sub(b"", content)
        if BRAND in remainder.lower():
            sys.stdout.buffer.write(path + b":" + number + b":" + content + b"\n")


if __name__ == "__main__":
    main()
