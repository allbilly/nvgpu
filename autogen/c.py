"""Small ctypes support layer used by the checked-in generated bindings."""
from __future__ import annotations

import ctypes, functools, os, pathlib, re, sys, sysconfig
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar, get_args

DEBUG = int(os.environ.get("DEBUG", "0"))
OSX, WIN = sys.platform == "darwin", os.name == "nt"

def getenv(key: str, default=""):
  value = os.environ.get(key)
  if value is None: return default
  try: return int(value)
  except ValueError: return value

def ceildiv(value: int, amount: int) -> int: return -(value // -amount)

def _do_ioctl(direction, base, number, struct_type, fd, *args, _payload=None, **kwargs):
  assert not WIN, "ioctl not supported"
  import fcntl
  ioctl = fd.ioctl if hasattr(fd, "ioctl") else functools.partial(fcntl.ioctl, fd)
  if struct_type is None: return ioctl((base << 8) | number, _payload or (args[0] if args else 0))
  output = _payload or struct_type(*args, **kwargs)
  request = (direction << 30) | (ctypes.sizeof(output) << 16) | (base << 8) | number
  if (rc := ioctl(request, output)): raise RuntimeError(f"ioctl returned {rc}")
  return output

def _base_value(base): return ord(base) if isinstance(base, str) else base
def _IO(base, number): return functools.partial(_do_ioctl, 0, _base_value(base), number, None)
def _IOW(base, number, typ): return functools.partial(_do_ioctl, 1, _base_value(base), number, typ)
def _IOR(base, number, typ): return functools.partial(_do_ioctl, 2, _base_value(base), number, typ)
def _IOWR(base, number, typ): return functools.partial(_do_ioctl, 3, _base_value(base), number, typ)

T, U, P = TypeVar("T"), TypeVar("U"), ParamSpec("P")

class POINTER(Generic[T], ctypes._Pointer):
  def __class_getitem__(cls, key): return ctypes.POINTER(key)

def pointer(value: T) -> POINTER[T]: return ctypes.pointer(value)  # type: ignore

if TYPE_CHECKING: _CFuncPtr = ctypes._CFunctionType
else: _CFuncPtr = ctypes._CFuncPtr

class CFUNCTYPE(Generic[T, P], _CFuncPtr):
  _flags_ = 0
  def __class_getitem__(cls, key): return ctypes.CFUNCTYPE(key[0], *key[1])

class Array(Generic[T, U], ctypes.Array):
  _type_, _length_ = ctypes.c_byte, 0
  def __class_getitem__(cls, key): return key[0] * get_args(key[1])[0]
  def __new__(cls, typ, length): return typ * length

class Struct(ctypes.Structure):
  SIZE = 0

  def __init__(self, *args, **kwargs):
    ctypes.Structure.__init__(self)
    for field, value in [*zip((item[0] for item in self._real_fields_), args), *kwargs.items()]: setattr(self, field, value)

  @classmethod
  def register_fields(cls, fields):
    setattr(cls, "_real_fields_", fields)
    for index, (name, *args) in enumerate(fields): setattr(cls, name, Field(*args, name=name, idx=index))

def record(cls) -> type[Struct]:
  setattr(cls, "_fields_", [("_mem_", ctypes.c_byte * cls.SIZE)])
  return cls

class Field:
  def __init__(self, typ, offset, bit_width=None, bit_offset=0, *, name=None, idx=0):
    self.typ, self.off, self.bit_width, self.bit_off = typ, offset, bit_width, bit_offset
    self.name, self.idx = name, idx

  def __set_name__(self, owner, name):
    entry = (name, self.typ, self.off) + ((self.bit_width, self.bit_off) if self.bit_width else ())
    if hasattr(owner, "_real_fields_"): owner._real_fields_.append(entry)
    else: setattr(owner, "_real_fields_", [entry])
    self.name, self.idx = name, len(owner._real_fields_) - 1

  def _resolve(self, cls):
    if self.bit_width:
      size = ceildiv(self.bit_width + self.bit_off, 8)
      field_slice, mask = slice(self.off, self.off + size), (1 << self.bit_width) - 1
      set_mask = ~(mask << self.bit_off)
      def bytes_to_int(obj): return int.from_bytes(memoryview(obj).cast("B")[field_slice], sys.byteorder)
      def set_bits(obj, value):
        result = (bytes_to_int(obj) & set_mask) | value << self.bit_off
        memoryview(obj).cast("B")[field_slice] = result.to_bytes(size, sys.byteorder)
      cfield = property(lambda obj: bytes_to_int(obj) >> self.bit_off & mask, set_bits)
    else:
      fields = [(str(index), ctypes.c_byte * 0) for index in range(self.idx)]
      fields += [("_", ctypes.c_byte * self.off), ("v", self.typ)]
      cfield = type(self.name, (ctypes.Structure,), {"_layout_": "ms", "_pack_": 1, "_fields_": fields}).v
    setattr(cls, self.name, cfield)
    return cfield

  def __get__(self, obj, objtype=None): return self._resolve(objtype).__get__(obj, objtype) if objtype else self
  def __set__(self, obj, value): self._resolve(obj.__class__).__set__(obj, value)

@functools.cache
def init_c_struct_t(size: int, fields: tuple[tuple, ...]):
  generated = type("CStruct", (Struct,), {"_fields_": [("_mem_", ctypes.c_byte * size)]})
  generated.register_fields(fields)
  return generated

def init_c_var(typ, create_cb):
  value = typ()
  create_cb(value)
  return value

class DLL(ctypes.CDLL):
  _loaded_: set[str] = set()

  @staticmethod
  def findlib(name: str, paths: list[str], extra_paths=[]):
    if name == "libc" and OSX: return "/usr/lib/libc.dylib"
    explicit = getenv(name.replace("-", "_").upper() + "_PATH", "")
    if explicit and pathlib.Path(explicit).is_file(): return explicit
    for path in paths:
      defaults = {
        "posix": [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item] + ["/usr/lib64", "/usr/lib", "/usr/local/lib"],
        "nt": os.environ["PATH"].split(os.pathsep),
        "darwin": ["/opt/homebrew/lib", f"/System/Library/Frameworks/{path}.framework", f"/System/Library/PrivateFrameworks/{path}.framework"],
        "linux": ["/lib", "/lib64", f"/lib/{sysconfig.get_config_var('MULTIARCH')}", "/usr/lib/wsl/lib/"],
      }
      candidate = pathlib.Path(path)
      if candidate.is_absolute():
        if candidate.is_file(): return path
        continue
      roots = ([explicit] if explicit else []) + defaults.get(os.name, []) + defaults.get(sys.platform, []) + extra_paths
      for prefix in map(pathlib.Path, roots):
        if not prefix.is_dir(): continue
        if WIN or OSX:
          names = [f"lib{path}.dylib", f"{path}.dylib", path] if OSX else [f"{path}.dll"]
          for basename in names:
            library = prefix / basename
            if library.is_file() or (OSX and "framework" in str(library) and library.is_symlink()): return str(library)
        else:
          for library in prefix.iterdir():
            if not library.is_file() or not re.fullmatch(f"lib{path}\\.so\\.?[0-9]*", library.name): continue
            with open(library, "rb") as stream:
              if stream.read(4) == b"\x7fELF": return str(library)

  def __init__(self, name: str, paths: str | list[str], extra_paths=[], emsg="", **kwargs):
    self.nm, self.emsg = name, emsg or f"try setting {name.upper() + '_PATH'}?"
    search_paths = paths if isinstance(paths, list) else [paths]
    search_extra = extra_paths if isinstance(extra_paths, list) else [extra_paths]
    if (path := DLL.findlib(name, search_paths, search_extra)):
      if DEBUG >= 3: print(f"loading {name} from {path}")
      try:
        super().__init__(path, **kwargs)
        self._loaded_.add(self.nm)
      except OSError as error:
        self.emsg = str(error)
        if DEBUG >= 3: print(f"loading {name} failed: {error}")
    elif DEBUG >= 3: print(f"loading {name} failed: not found on system")

  def bind(self, restype, *argtypes):
    def decorate(function):
      cfunc = None
      @functools.wraps(function)
      def wrapper(*args):
        nonlocal cfunc
        if cfunc is None:
          cfunc = getattr(self, function.__name__)
          cfunc.argtypes, cfunc.restype = argtypes, restype
        return cfunc(*args)
      return wrapper
    return decorate

  def __getattr__(self, name):
    if self.nm not in self._loaded_: raise AttributeError(f"failed to load library {self.nm}: {self.emsg}")
    return super().__getattr__(name)
