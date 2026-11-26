"""Keep git-grep hits with brand mentions outside the two places that may name it.

Those are the project's public repository URLs, in any file, and the copyright
holder at the start of a notice line in the root LICENSE.
"""

import re
import sys

BRAND = b"startup" + b"bros"
PUBLIC_REPOSITORY = re.compile(
    rb"(?:^|(?<=[\s`\"'(<\[{]))(?:git\+)?(?:https://)?github\.com/"
    + re.escape(BRAND)
    + rb"(?:-com)?/synthetic-panel(?:\.git)?(?=$|[\s/@?#`\"'<>()\[\]{},;])",
    re.IGNORECASE,
)
# A license has to say who holds the copyright, so the root LICENSE may name the
# company there and nowhere else. Only the holder at the start of a notice line
# is removed. Whatever follows it on that line, every other line of the file and
# every other file, a LICENSE in a subdirectory included, are still checked.
LICENSE_PATH = b"LICENSE"
COPYRIGHT_HOLDER = re.compile(
    rb"^Copyright \(c\) \d{4}(?:\s*[-,]\s*\d{4})* "
    + re.escape(BRAND)
    + rb" LLC(?=$|[\s,.(])",
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
        remainder = content
        if path == LICENSE_PATH:
            remainder = COPYRIGHT_HOLDER.sub(b"", remainder)
        remainder = PUBLIC_REPOSITORY.sub(b"", remainder)
        if BRAND in remainder.lower():
            sys.stdout.buffer.write(path + b":" + number + b":" + content + b"\n")


if __name__ == "__main__":
    main()
