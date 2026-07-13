import ast
import struct
import sys
from pathlib import Path


def read_npy_header(path):
    with open(path, "rb") as f:
        magic = f.read(6)

        if magic != b"\x93NUMPY":
            raise ValueError("Arquivo não parece ser um .npy válido.")

        major, minor = struct.unpack("BB", f.read(2))

        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"Versão .npy não suportada: {major}.{minor}")

        header = f.read(header_len)

        if major == 3:
            header = header.decode("utf-8")
        else:
            header = header.decode("latin1")

        header_dict = ast.literal_eval(header)

        return major, minor, header_dict


def main():
    if len(sys.argv) != 2:
        print(f"Uso: python3 {sys.argv[0]} arquivo.npy")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}")
        sys.exit(1)

    major, minor, header = read_npy_header(path)

    print(f"arquivo:       {path}")
    print(f"versao npy:    {major}.{minor}")
    print(f"shape:         {header['shape']}")
    print(f"dtype:         {header['descr']}")
    print(f"fortran_order: {header['fortran_order']}")


if __name__ == "__main__":
    main()
