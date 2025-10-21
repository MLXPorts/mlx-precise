// Copyright © 2023-2024 Apple Inc.

#include <nanobind/stl/vector.h>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

bool is_str_or_path(nb::object obj) {
  if (nb::isinstance<nb::str>(obj)) {
    return true;
  }
  nb::object path_type = nb::module_::import_("pathlib").attr("Path");
  return nb::isinstance(obj, path_type);
}

bool is_istream_object(const nb::object& file) {
  return nb::hasattr(file, "readinto") && nb::hasattr(file, "seek") &&
      nb::hasattr(file, "tell") && nb::hasattr(file, "closed");
}

bool is_ostream_object(const nb::object& file) {
  return nb::hasattr(file, "write") && nb::hasattr(file, "seek") &&
      nb::hasattr(file, "tell") && nb::hasattr(file, "closed");
}

bool is_zip_file(const nb::module_& zipfile, const nb::object& file) {
  if (is_istream_object(file)) {
    auto st_pos = file.attr("tell")();
    bool r = nb::cast<bool>(zipfile.attr("is_zipfile")(file));
    file.attr("seek")(st_pos, 0);
    return r;
  }
  return nb::cast<bool>(zipfile.attr("is_zipfile")(file));
}

class ZipFileWrapper {
 public:
  ZipFileWrapper(
      const nb::module_& zipfile,
      const nb::object& file,
      char mode = 'r',
      int compression = 0)
      : zipfile_module_(zipfile),
        zipfile_object_(zipfile.attr("ZipFile")(
            file,
            "mode"_a = mode,
            "compression"_a = compression,
            "allowZip64"_a = true)),
        files_list_(zipfile_object_.attr("namelist")()),
        open_func_(zipfile_object_.attr("open")),
        read_func_(zipfile_object_.attr("read")),
        close_func_(zipfile_object_.attr("close")) {}

  std::vector<std::string> namelist() const {
    return nb::cast<std::vector<std::string>>(files_list_);
  }

  nb::object open(const std::string& key, char mode = 'r') {
    // Following numpy :
    // https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/npyio.py#L742C36-L742C47
    if (mode == 'w') {
      return open_func_(key, "mode"_a = mode, "force_zip64"_a = true);
    }
    return open_func_(key, "mode"_a = mode);
  }

 private:
  nb::module_ zipfile_module_;
  nb::object zipfile_object_;
  nb::list files_list_;
  nb::object open_func_;
  nb::object read_func_;
  nb::object close_func_;
};

///////////////////////////////////////////////////////////////////////////////
// Loading
///////////////////////////////////////////////////////////////////////////////

class PyFileReader : public mx::io::Reader {
 public:
  PyFileReader(nb::object file)
      : pyistream_(file),
        readinto_func_(file.attr("readinto")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileReader() {
    nb::gil_scoped_acquire gil;

    pyistream_.release().dec_ref();
    readinto_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !nb::cast<bool>(pyistream_.attr("closed"));
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !pyistream_.is_none();
    }
    return out;
  }

  size_t tell() override {
    size_t out;
    {
      nb::gil_scoped_acquire gil;
      out = nb::cast<size_t>(tell_func_());
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    nb::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void read(char* data, size_t n) override {
    nb::gil_scoped_acquire gil;
    _read(data, n);
  }

  void read(char* data, size_t n, size_t offset) override {
    nb::gil_scoped_acquire gil;
    seek_func_(offset, (int)std::ios_base::beg);
    _read(data, n);
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  void _read(char* data, size_t n) {
    auto memview = PyMemoryView_FromMemory(data, n, PyBUF_WRITE);
    nb::object bytes_read = readinto_func_(nb::handle(memview));

    if (bytes_read.is_none() || nb::cast<size_t>(bytes_read) < n) {
      throw std::runtime_error("[load] Failed to read from python stream");
    }
  }

  nb::object pyistream_;
  nb::object readinto_func_;
  nb::object seek_func_;
  nb::object tell_func_;
};

std::pair<
    std::unordered_map<std::string, mx::array>,
    std::unordered_map<std::string, std::string>>
mlx_load_safetensor_helper(nb::object file, mx::StreamOrDevice s) {
  if (is_str_or_path(file)) { // Assume .safetensors file path string
    auto file_str = nb::cast<std::string>(nb::str(file));
    return mx::load_safetensors(file_str, s);
  } else if (is_istream_object(file)) {
    // If we don't own the stream and it was passed to us, eval immediately
    auto res = mx::load_safetensors(std::make_shared<PyFileReader>(file), s);
    {
      nb::gil_scoped_release gil;
      for (auto& [key, arr] : std::get<0>(res)) {
        arr.eval();
      }
    }
    return res;
  }

  throw std::invalid_argument(
      "[load_safetensors] Input must be a file-like object, or string");
}

mx::GGUFLoad mlx_load_gguf_helper(nb::object file, mx::StreamOrDevice s) {
  if (is_str_or_path(file)) { // Assume .gguf file path string
    auto file_str = nb::cast<std::string>(nb::str(file));
    return mx::load_gguf(file_str, s);
  }

  throw std::invalid_argument("[load_gguf] Input must be a string");
}

// Minimal NPZ reader (store-only). If any entry is compressed (method=8), raise.
std::unordered_map<std::string, mx::array> mlx_load_npz_helper(
    nb::object file,
    mx::StreamOrDevice s) {
  auto read_all = [&](nb::object f) -> std::vector<char> {
    // Use Python file API to read all bytes
    auto st_pos = f.attr("tell")();
    f.attr("seek")(0, 2);
    size_t size = nb::cast<size_t>(f.attr("tell")());
    f.attr("seek")(0, 0);
    nb::bytes b = nb::cast<nb::bytes>(f.attr("read")(size));
    // restore
    f.attr("seek")(st_pos, 0);
    return std::vector<char>(b.c_str(), b.c_str() + PyBytes_GET_SIZE(b.ptr()));
  };

  std::vector<char> buf;
  if (is_str_or_path(file)) {
    std::ifstream ifs(nb::cast<std::string>(nb::str(file)), std::ios::binary);
    if (!ifs) throw std::runtime_error("[load_npz] failed to open file");
    ifs.seekg(0, std::ios::end);
    size_t size = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    buf.resize(size);
    ifs.read(buf.data(), size);
  } else if (is_istream_object(file)) {
    buf = read_all(file);
  } else {
    throw std::invalid_argument("[load_npz] Input must be a file-like object, or string");
  }

  auto rd32 = [&](size_t off) -> uint32_t {
    const unsigned char* p = (const unsigned char*)buf.data() + off;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
  };
  auto rd16 = [&](size_t off) -> uint16_t {
    const unsigned char* p = (const unsigned char*)buf.data() + off;
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  };

  // Find EOCD signature 0x06054b50 scanning last 64KB
  size_t n = buf.size();
  size_t start = (n > 66000 ? n - 66000 : 0);
  size_t eocd = n;
  for (size_t i = n >= 22 ? n - 22 : 0; i-- > start;) {
    if (rd32(i) == 0x06054B50u) { eocd = i; break; }
  }
  if (eocd == n) {
    throw std::runtime_error("[load_npz] EOCD not found; not a ZIP file");
  }
  uint16_t total_entries = rd16(eocd + 10);
  uint32_t cd_size = rd32(eocd + 12);
  uint32_t cd_offset = rd32(eocd + 16);
  if (cd_offset + cd_size > n) {
    throw std::runtime_error("[load_npz] Central directory out of bounds");
  }

  std::unordered_map<std::string, mx::array> out;
  size_t p = cd_offset;
  for (uint16_t i = 0; i < total_entries; ++i) {
    if (rd32(p) != 0x02014B50u) throw std::runtime_error("[load_npz] Bad CEN sig");
    uint16_t method = rd16(p + 10);
    uint32_t crc = rd32(p + 16);
    uint32_t csize = rd32(p + 20);
    uint32_t usize = rd32(p + 24);
    uint16_t nlen = rd16(p + 28);
    uint16_t elen = rd16(p + 30);
    uint16_t clen = rd16(p + 32);
    uint32_t lhoff = rd32(p + 42);
    std::string name(buf.data() + p + 46, buf.data() + p + 46 + nlen);
    p += 46 + nlen + elen + clen;

    if (method != 0 && method != 8) {
      throw std::runtime_error("[load_npz] Unsupported ZIP method");
    }
    // Read local header
    if (rd32(lhoff) != 0x04034B50u) throw std::runtime_error("[load_npz] Bad LOC sig");
    uint16_t nlen2 = rd16(lhoff + 26);
    uint16_t elen2 = rd16(lhoff + 28);
    size_t data_off = lhoff + 30 + nlen2 + elen2;
    if (data_off + csize > n) throw std::runtime_error("[load_npz] data OOB");
    const char* data = buf.data() + data_off;
    std::vector<char> payload;
#ifdef MLX_HAVE_ZLIB
    if (method == 8) {
      // Inflate using zlib
      payload.resize(usize);
      uLongf dest_len = (uLongf)usize;
      int zrc = uncompress((Bytef*)payload.data(), &dest_len, (const Bytef*)data, (uLongf)csize);
      if (zrc != Z_OK || dest_len != (uLongf)usize) {
        throw std::runtime_error("[load_npz] zlib inflate failed");
      }
      data = payload.data();
      csize = usize;
    } else
#endif
    {
      // method==0 store; nothing to do
    }

    // Wrap in a temporary Python bytes-backed reader: simplest path is to use
    // a Python memoryview via PyFileReader -> but we can directly build using
    // the existing core .npy loader by creating a temporary Reader over memory.
    class MemReader : public mx::io::Reader {
     public:
      MemReader(const char* d, size_t n) : d_(d), n_(n), pos_(0) {}
      bool is_open() const override { return true; }
      bool good() const override { return true; }
      size_t tell() override { return pos_; }
      void seek(int64_t off, std::ios_base::seekdir way) override {
        if (way == std::ios_base::beg) pos_ = (size_t)off; else pos_ += off;
      }
      void read(char* out, size_t n) override {
        if (pos_ + n > n_) throw std::runtime_error("[npz] read OOB");
        std::memcpy(out, d_ + pos_, n); pos_ += n;
      }
      void read(char* out, size_t n, size_t off) override {
        if (off + n > n_) throw std::runtime_error("[npz] read OOB");
        std::memcpy(out, d_ + off, n);
      }
      std::string label() const override { return "npz memory"; }
     private:
      const char* d_;
      size_t n_;
      size_t pos_;
    } mr(data, csize);

    auto arr = mx::load(std::make_shared<MemReader>(mr), s);
    std::string key = name;
    if (key.size() > 4 && key.substr(key.size()-4) == ".npy") key.resize(key.size()-4);
    out.emplace(std::move(key), std::move(arr));
  }

  return out;
}

mx::array mlx_load_npy_helper(nb::object file, mx::StreamOrDevice s) {
  if (is_str_or_path(file)) { // Assume .npy file path string
    auto file_str = nb::cast<std::string>(nb::str(file));
    return mx::load(file_str, s);
  } else if (is_istream_object(file)) {
    // If we don't own the stream and it was passed to us, eval immediately
    auto arr = mx::load(std::make_shared<PyFileReader>(file), s);
    {
      nb::gil_scoped_release gil;
      arr.eval();
    }
    return arr;
  }
  throw std::invalid_argument(
      "[load_npy] Input must be a file-like object, or string");
}

LoadOutputTypes mlx_load_helper(
    nb::object file,
    std::optional<std::string> format,
    bool return_metadata,
    mx::StreamOrDevice s) {
  if (!format.has_value()) {
    std::string fname;
    if (is_str_or_path(file)) {
      fname = nb::cast<std::string>(nb::str(file));
    } else if (is_istream_object(file)) {
      fname = nb::cast<std::string>(file.attr("name"));
    } else {
      throw std::invalid_argument(
          "[load] Input must be a file-like object opened in binary mode, or string");
    }
    size_t ext = fname.find_last_of('.');
    if (ext == std::string::npos) {
      throw std::invalid_argument(
          "[load] Could not infer file format from extension");
    }
    format.emplace(fname.substr(ext + 1));
  }

  if (return_metadata && (format.value() == "npy" || format.value() == "npz")) {
    throw std::invalid_argument(
        "[load] metadata not supported for format " + format.value());
  }
  if (format.value() == "safetensors") {
    auto [dict, metadata] = mlx_load_safetensor_helper(file, s);
    if (return_metadata) {
      return std::make_pair(dict, metadata);
    }
    return dict;
  } else if (format.value() == "npz") {
    return mlx_load_npz_helper(file, s);
  } else if (format.value() == "npy") {
    return mlx_load_npy_helper(file, s);
  } else if (format.value() == "gguf") {
    auto [weights, metadata] = mlx_load_gguf_helper(file, s);
    if (return_metadata) {
      return std::make_pair(weights, metadata);
    } else {
      return weights;
    }
  } else {
    throw std::invalid_argument("[load] Unknown file format " + format.value());
  }
}

///////////////////////////////////////////////////////////////////////////////
// Saving
///////////////////////////////////////////////////////////////////////////////

class PyFileWriter : public mx::io::Writer {
 public:
  PyFileWriter(nb::object file)
      : pyostream_(file),
        write_func_(file.attr("write")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileWriter() {
    nb::gil_scoped_acquire gil;

    pyostream_.release().dec_ref();
    write_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !nb::cast<bool>(pyostream_.attr("closed"));
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !pyostream_.is_none();
    }
    return out;
  }

  size_t tell() override {
    size_t out;
    {
      nb::gil_scoped_acquire gil;
      out = nb::cast<size_t>(tell_func_());
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    nb::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void write(const char* data, size_t n) override {
    nb::gil_scoped_acquire gil;

    auto memview =
        PyMemoryView_FromMemory(const_cast<char*>(data), n, PyBUF_READ);
    nb::object bytes_written = write_func_(nb::handle(memview));

    if (bytes_written.is_none() || nb::cast<size_t>(bytes_written) < n) {
      throw std::runtime_error("[load] Failed to write to python stream");
    }
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  nb::object pyostream_;
  nb::object write_func_;
  nb::object seek_func_;
  nb::object tell_func_;
};

void mlx_save_helper(nb::object file, mx::array a) {
  if (is_str_or_path(file)) {
    auto file_str = nb::cast<std::string>(nb::str(file));
    mx::save(file_str, a);
    return;
  } else if (is_ostream_object(file)) {
    auto writer = std::make_shared<PyFileWriter>(file);
    {
      nb::gil_scoped_release gil;
      mx::save(writer, a);
    }

    return;
  }

  throw std::invalid_argument(
      "[save] Input must be a file-like object, or string");
}

void mlx_savez_helper(
    nb::object file_,
    nb::args args,
    const nb::kwargs& kwargs,
    bool compressed) {
  // Add .npz to the end of the filename if not already there
  nb::object file = file_;

  if (is_str_or_path(file)) {
    std::string fname = nb::cast<std::string>(nb::str(file_));

    // Add .npz to file name if it is not there
    if (fname.length() < 4 || fname.substr(fname.length() - 4, 4) != ".npz")
      fname += ".npz";

    file = nb::cast(fname);
  }

  // Collect args and kwargs
  auto arrays_dict =
      nb::cast<std::unordered_map<std::string, mx::array>>(kwargs);
  auto arrays_list = nb::cast<std::vector<mx::array>>(args);

  for (int i = 0; i < arrays_list.size(); i++) {
    std::string arr_name = "arr_" + std::to_string(i);

    if (arrays_dict.count(arr_name) > 0) {
      throw std::invalid_argument(
          "[savez] Cannot use un-named variables and keyword " + arr_name);
    }

    arrays_dict.insert({arr_name, arrays_list[i]});
  }

  // Use C++ core NPZ writer (store-only for now)
  if (is_str_or_path(file)) {
    auto file_str = nb::cast<std::string>(nb::str(file));
    nb::gil_scoped_release nogil;
    mx::savez(file_str, arrays_dict, /*compressed=*/compressed);
    return;
  } else if (is_ostream_object(file)) {
    auto writer = std::make_shared<PyFileWriter>(file);
    nb::gil_scoped_release nogil;
    mx::savez(writer, arrays_dict, /*compressed=*/compressed);
    return;
  }
  throw std::invalid_argument("[savez] Input must be a file-like object, or string");
}

void mlx_save_safetensor_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m) {
  std::unordered_map<std::string, std::string> metadata_map;
  if (m) {
    try {
      metadata_map =
          nb::cast<std::unordered_map<std::string, std::string>>(m.value());
    } catch (const nb::cast_error& e) {
      throw std::invalid_argument(
          "[save_safetensors] Metadata must be a dictionary with string keys and values");
    }
  } else {
    metadata_map = std::unordered_map<std::string, std::string>();
  }
  auto arrays_map = nb::cast<std::unordered_map<std::string, mx::array>>(d);
  if (is_str_or_path(file)) {
    {
      auto file_str = nb::cast<std::string>(nb::str(file));
      nb::gil_scoped_release nogil;
      mx::save_safetensors(file_str, arrays_map, metadata_map);
    }
  } else if (is_ostream_object(file)) {
    auto writer = std::make_shared<PyFileWriter>(file);
    {
      nb::gil_scoped_release nogil;
      mx::save_safetensors(writer, arrays_map, metadata_map);
    }
  } else {
    throw std::invalid_argument(
        "[save_safetensors] Input must be a file-like object, or string");
  }
}

void mlx_save_gguf_helper(
    nb::object file,
    nb::dict a,
    std::optional<nb::dict> m) {
  auto arrays_map = nb::cast<std::unordered_map<std::string, mx::array>>(a);
  if (is_str_or_path(file)) {
    if (m) {
      auto metadata_map =
          nb::cast<std::unordered_map<std::string, mx::GGUFMetaData>>(
              m.value());
      {
        auto file_str = nb::cast<std::string>(nb::str(file));
        nb::gil_scoped_release nogil;
        mx::save_gguf(file_str, arrays_map, metadata_map);
      }
    } else {
      {
        auto file_str = nb::cast<std::string>(nb::str(file));
        nb::gil_scoped_release nogil;
        mx::save_gguf(file_str, arrays_map);
      }
    }
  } else {
    throw std::invalid_argument("[save_gguf] Input must be a string");
  }
}
