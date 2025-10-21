// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

// Used by pread implementation.
#ifdef _WIN32
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <windows.h>
#endif // _WIN32

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"
#ifdef MLX_HAVE_ZLIB
#include <zlib.h>
#endif

// Adapted from
// https://github.com/angeloskath/supervised-lda/blob/master/include/ldaplusplus/NumpyFormat.hpp

namespace mlx::core {

namespace {

constexpr uint8_t MAGIC[] = {
    0x93,
    0x4e,
    0x55,
    0x4d,
    0x50,
    0x59,
};

inline bool is_big_endian() {
  union ByteOrder {
    int32_t i;
    uint8_t c[4];
  };
  ByteOrder b = {0x01234567};

  return b.c[0] == 0x01;
}

// Array protocol typestring for Dtype
std::string dtype_to_array_protocol(const Dtype& t) {
  std::ostringstream r;
  if (size_of(t) > 1) {
    r << (is_big_endian() ? ">" : "<");
  } else {
    r << "|";
  }
  r << kindof(t) << (int)size_of(t);
  return r.str();
}

// Dtype from array protocol type string
Dtype dtype_from_array_protocol(std::string_view t) {
  if (t.length() == 2 || t.length() == 3) {
    std::string_view r = t.length() == 3 ? t.substr(1, 2) : t;

    if (r == "V2") {
      return bfloat16;
    }

    uint8_t size = r[1] - '0';

    switch (r[0]) {
      case 'b': {
        if (size == 1)
          return bool_;
      }
      case 'i': {
        if (size == 1)
          return int8;
        else if (size == 2)
          return int16;
        else if (size == 4)
          return int32;
        else if (size == 8)
          return int64;
      }
      case 'u': {
        if (size == 1)
          return uint8;
        else if (size == 2)
          return uint16;
        else if (size == 4)
          return uint32;
        else if (size == 8)
          return uint64;
      }
      case 'f': {
        if (size == 2)
          return float16;
        else if (size == 4)
          return float32;
      }
      case 'c': {
        return complex64;
      }
    }
  }

  throw std::invalid_argument(
      "[from_str] Invalid array protocol type-string: " + std::string(t));
}

#ifdef _WIN32
// There is no pread on Windows, emulate it with ReadFile.
int64_t pread(int fd, void* buf, uint64_t size, uint64_t offset) {
  HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE) {
    return -1;
  }

  OVERLAPPED overlapped = {0};
  overlapped.Offset = offset & 0xFFFFFFFF;
  overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;

  DWORD bytes_read;
  if (!ReadFile(file, buf, size, &bytes_read, &overlapped)) {
    if (GetLastError() != ERROR_HANDLE_EOF) {
      return -1;
    }
  }

  return bytes_read;
}
#endif

} // namespace

/** Save array to out stream in .npy format */
void save(std::shared_ptr<io::Writer> out_stream, array a) {
  ////////////////////////////////////////////////////////
  // Check array

  a = contiguous(a, true);
  a.eval();

  if (a.nbytes() == 0) {
    throw std::invalid_argument("[save] cannot serialize an empty array");
  }

  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error("[save] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Prepare header
  std::ostringstream magic_ver_len;
  magic_ver_len.write(reinterpret_cast<const char*>(MAGIC), 6);

  std::string fortran_order = a.flags().col_contiguous ? "True" : "False";
  std::ostringstream header;
  header << "{'descr': '" << dtype_to_array_protocol(a.dtype()) << "',"
         << " 'fortran_order': " << fortran_order << "," << " 'shape': (";
  for (auto i : a.shape()) {
    header << i << ", ";
  }
  header << ")}";

  size_t header_len = static_cast<size_t>(header.tellp());
  bool is_v1 = header_len + 15 < std::numeric_limits<uint16_t>::max();

  // Pad out magic + version + header_len + header + \n to be divisible by 16
  size_t padding = (6 + 2 + (2 + 2 * is_v1) + header_len + 1) % 16;

  header << std::string(padding, ' ') << '\n';

  if (is_v1) {
    magic_ver_len << (char)0x01 << (char)0x00;

    uint16_t v1_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v1_header_len);

    if (!is_big_endian()) {
      magic_ver_len.write(len_bytes, 2);
    } else {
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  } else {
    magic_ver_len << (char)0x02 << (char)0x00;

    uint32_t v2_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v2_header_len);

    if (!is_big_endian()) {
      magic_ver_len.write(len_bytes, 4);
    } else {
      magic_ver_len.write(len_bytes + 3, 1);
      magic_ver_len.write(len_bytes + 2, 1);
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  }
  ////////////////////////////////////////////////////////
  // Serialize array

  out_stream->write(magic_ver_len.str().c_str(), magic_ver_len.str().length());
  out_stream->write(header.str().c_str(), header.str().length());
  out_stream->write(a.data<char>(), a.nbytes());
}

/** Save array to file in .npy format */
void save(std::string file, array a) {
  // Add .npy to file name if it is not there
  if (file.length() < 4 || file.substr(file.length() - 4, 4) != ".npy")
    file += ".npy";

  // Serialize array
  save(std::make_shared<io::FileWriter>(std::move(file)), a);
}

/** Load array from reader in .npy format */
array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error("[load] Failed to open " + in_stream->label());
  }

  auto stream = to_stream(s, Device::cpu);
  if (stream.device != Device::cpu) {
    throw std::runtime_error("[load] Must run on a CPU stream.");
  }

  ////////////////////////////////////////////////////////
  // Read header and prepare array details

  // Read and check magic
  char read_magic_and_ver[8];
  in_stream->read(read_magic_and_ver, 8);
  if (std::memcmp(read_magic_and_ver, MAGIC, 6) != 0) {
    throw std::runtime_error("[load] Invalid header in " + in_stream->label());
  }

  // Read and check version
  if (read_magic_and_ver[6] != 1 && read_magic_and_ver[6] != 2) {
    throw std::runtime_error(
        "[load] Unsupported npy format version in " + in_stream->label());
  }

  // Read header len and header
  int header_len_size = read_magic_and_ver[6] == 1 ? 2 : 4;
  size_t header_len;

  if (header_len_size == 2) {
    uint16_t v1_header_len;
    in_stream->read(reinterpret_cast<char*>(&v1_header_len), header_len_size);
    header_len = v1_header_len;
  } else {
    uint32_t v2_header_len;
    in_stream->read(reinterpret_cast<char*>(&v2_header_len), header_len_size);
    header_len = v2_header_len;
  }

  // Read the header
  std::vector<char> buffer(header_len + 1);
  in_stream->read(&buffer[0], header_len);
  buffer[header_len] = 0;
  std::string header(&buffer[0]);

  // Read data type from header
  std::string dtype_str = header.substr(11, 3);
  bool read_is_big_endian = dtype_str[0] == '>';
  Dtype dtype = dtype_from_array_protocol(dtype_str);

  // Read contiguity order
  bool col_contiguous = header[34] == 'T';

  // Read array shape from header
  Shape shape;

  size_t st = header.find_last_of('(') + 1;
  size_t ed = header.find_last_of(')');
  std::string shape_str = header.substr(st, ed - st);

  while (!shape_str.empty()) {
    // Read current number and get position of comma
    size_t pos;
    int dim = std::stoi(shape_str, &pos);
    shape.push_back(dim);

    // Skip the comma and space and read the next number
    if (pos + 2 <= shape_str.length())
      shape_str = shape_str.substr(pos + 2);
    else {
      shape_str = shape_str.substr(pos);
      if (!shape_str.empty() && shape_str != " " && shape_str != ",") {
        throw std::runtime_error(
            "[load] Unknown error while parsing header in " +
            in_stream->label());
      }
      shape_str = "";
    }
  }

  ////////////////////////////////////////////////////////
  // Build primitive

  size_t offset = 8 + header_len_size + header.length();
  bool swap_endianness = read_is_big_endian != is_big_endian();

  if (col_contiguous) {
    std::reverse(shape.begin(), shape.end());
  }
  auto loaded_array = array(
      shape,
      dtype,
      std::make_shared<Load>(stream, in_stream, offset, swap_endianness),
      std::vector<array>{});
  if (col_contiguous) {
    loaded_array = transpose(loaded_array, s);
  }

  return loaded_array;
}

/** Load array from file in .npy format */
array load(std::string file, StreamOrDevice s) {
  return load(std::make_shared<io::ParallelFileReader>(std::move(file)), s);
}

namespace io {

ThreadPool& thread_pool() {
  static ThreadPool pool_{4};
  return pool_;
}

ThreadPool& ParallelFileReader::thread_pool() {
  static ThreadPool thread_pool{4};
  return thread_pool;
}

void ParallelFileReader::read(char* data, size_t n) {
  while (n != 0) {
    auto m = ::read(fd_, data, std::min(n, static_cast<size_t>(INT32_MAX)));
    if (m <= 0) {
      std::ostringstream msg;
      msg << "[read] Unable to read " << n << " bytes from file.";
      throw std::runtime_error(msg.str());
    }
    data += m;
    n -= m;
  }
}

void ParallelFileReader::read(char* data, size_t n, size_t offset) {
  auto readfn = [fd = fd_](size_t offset, size_t size, char* buffer) -> bool {
    while (size != 0) {
      auto m = pread(fd, buffer, size, offset);
      if (m <= 0) {
        return false;
      }
      buffer += m;
      size -= m;
    }
    return true;
  };
  std::vector<std::future<bool>> futs;
  while (n != 0) {
    if (n < batch_size_) {
      if (!readfn(offset, n, data)) {
        throw std::runtime_error("[read] Unable to read from file.");
      }
      break;
    } else {
      size_t m = batch_size_;
      futs.emplace_back(
          ParallelFileReader::thread_pool().enqueue(readfn, offset, m, data));
      data += m;
      n -= m;
      offset += m;
    }
  }
  for (auto& f : futs) {
    if (!f.get()) {
      throw std::runtime_error("[read] Unable to read from file.");
    }
  }
}

} // namespace io

} // namespace mlx::core

///////////////////////////////////////////////////////////////////////////////
// NPZ (ZIP of NPY) writer — store-only (no compression)
///////////////////////////////////////////////////////////////////////////////

namespace mlx::core {
namespace {

struct ZipEntry {
  std::string name;
  uint32_t crc32;
  uint32_t comp_size;
  uint32_t uncomp_size;
  uint32_t local_header_offset;
};

// Simple CRC32 (IEEE 802.3) implementation
static uint32_t crc32_update(uint32_t crc, const unsigned char* buf, size_t len) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int j = 0; j < 8; ++j) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[i] = c;
    }
    init = true;
  }
  crc = crc ^ 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

class MemoryWriter : public io::Writer {
 public:
  bool is_open() const override { return true; }
  bool good() const override { return true; }
  size_t tell() override { return buf_.size(); }
  void seek(int64_t off, std::ios_base::seekdir way) override {
    // Not supported; used only as sink for NPY serialization
    if (off != 0 || way != std::ios_base::beg) {
      throw std::runtime_error("MemoryWriter.seek not supported");
    }
  }
  void write(const char* data, size_t n) override {
    buf_.insert(buf_.end(), data, data + n);
  }
  std::string label() const override { return "memory buffer"; }
  const std::vector<char>& data() const { return buf_; }

 private:
  std::vector<char> buf_;
};

// Little-endian write helpers
inline void w16(std::vector<char>& out, uint16_t v) {
  out.push_back((char)(v & 0xFF));
  out.push_back((char)((v >> 8) & 0xFF));
}
inline void w32(std::vector<char>& out, uint32_t v) {
  out.push_back((char)(v & 0xFF));
  out.push_back((char)((v >> 8) & 0xFF));
  out.push_back((char)((v >> 16) & 0xFF));
  out.push_back((char)((v >> 24) & 0xFF));
}

void write_local_header(
    io::Writer& out,
    const ZipEntry& e,
    uint16_t method,
    std::vector<char>& scratch) {
  scratch.clear();
  // Local file header signature
  w32(scratch, 0x04034B50u);
  w16(scratch, 20);          // version needed
  w16(scratch, 0);           // flags
  w16(scratch, method);      // method: 0 = store, 8 = deflate
  w16(scratch, 0);           // time
  w16(scratch, 0);           // date
  w32(scratch, e.crc32);     // CRC-32
  w32(scratch, e.comp_size); // compressed size
  w32(scratch, e.uncomp_size); // uncompressed size
  w16(scratch, (uint16_t)e.name.size()); // file name length
  w16(scratch, 0);           // extra length
  out.write(scratch.data(), scratch.size());
  out.write(e.name.data(), e.name.size());
}

void write_central_header(
    std::vector<char>& cd,
    const ZipEntry& e,
    uint16_t method) {
  // Central dir file header signature
  w32(cd, 0x02014B50u);
  w16(cd, 20); // version made by
  w16(cd, 20); // version needed
  w16(cd, 0);  // flags
  w16(cd, method);  // method
  w16(cd, 0);  // time
  w16(cd, 0);  // date
  w32(cd, e.crc32);
  w32(cd, e.comp_size);
  w32(cd, e.uncomp_size);
  w16(cd, (uint16_t)e.name.size()); // file name length
  w16(cd, 0);  // extra length
  w16(cd, 0);  // comment length
  w16(cd, 0);  // disk number start
  w16(cd, 0);  // internal attrs
  w32(cd, 0);  // external attrs
  w32(cd, e.local_header_offset);
  // filename
  cd.insert(cd.end(), e.name.begin(), e.name.end());
}

void write_end_of_central(
    io::Writer& out,
    uint16_t nfiles,
    uint32_t cd_size,
    uint32_t cd_offset,
    std::vector<char>& scratch) {
  scratch.clear();
  w32(scratch, 0x06054B50u);
  w16(scratch, 0); // disk
  w16(scratch, 0); // start disk
  w16(scratch, nfiles);
  w16(scratch, nfiles);
  w32(scratch, cd_size);
  w32(scratch, cd_offset);
  w16(scratch, 0); // comment length
  out.write(scratch.data(), scratch.size());
}

} // anonymous namespace

void savez(
    std::shared_ptr<io::Writer> out_stream,
    const std::unordered_map<std::string, array>& arrays,
    bool compressed) {
  if (!out_stream || !out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error("[savez] Output stream not open");
  }

  std::vector<ZipEntry> entries;
  entries.reserve(arrays.size());

  std::vector<char> scratch; // reused buffer for headers
  std::vector<char> central_dir;

  // Write each local header + data
  for (const auto& [name, arr] : arrays) {
    std::string fname = name;
    if (fname.rfind(".npy") == std::string::npos) {
      fname += ".npy";
    }

    // Serialize .npy into memory to compute crc/size
    MemoryWriter mw;
    {
      array a = arr;
      a.eval();
      save(std::shared_ptr<io::Writer>(&mw, [](io::Writer*) {}), a);
    }
    const auto& data = mw.data();
    uint32_t crc = crc32_update(0, (const unsigned char*)data.data(), data.size());

    ZipEntry e;
    e.name = fname;
    e.crc32 = crc;
    e.uncomp_size = (uint32_t)data.size();
    e.local_header_offset = (uint32_t)out_stream->tell();

    uint16_t method = 0;
    if (compressed) {
      uLongf dest_len = compressBound((uLongf)data.size());
      std::vector<unsigned char> zbuf(dest_len);
      int zrc = compress2(zbuf.data(), &dest_len, (const Bytef*)data.data(), (uLongf)data.size(), Z_BEST_SPEED);
      if (zrc != Z_OK) throw std::runtime_error("[savez] zlib compress failed");
      zbuf.resize(dest_len);
      e.comp_size = (uint32_t)zbuf.size();
      method = 8;
      write_local_header(*out_stream, e, method, scratch);
      out_stream->write((const char*)zbuf.data(), zbuf.size());
    } else
    {
      e.comp_size = (uint32_t)data.size();
      method = 0;
      write_local_header(*out_stream, e, method, scratch);
      out_stream->write(data.data(), data.size());
    }

    entries.push_back(e);
  }

  // Build central directory in memory
  uint32_t cd_offset = (uint32_t)out_stream->tell();
  for (const auto& e : entries) {
    uint16_t method = (e.comp_size != e.uncomp_size) ? 8 : 0;
    write_central_header(central_dir, e, method);
  }
  // Write central directory
  if (!central_dir.empty()) {
    out_stream->write(central_dir.data(), central_dir.size());
  }
  uint32_t cd_size = (uint32_t)central_dir.size();
  write_end_of_central(*out_stream, (uint16_t)entries.size(), cd_size, cd_offset, scratch);
}

void savez(
    std::string file,
    const std::unordered_map<std::string, array>& arrays,
    bool compressed) {
  // Add .npz extension if missing
  if (file.length() < 4 || file.substr(file.length() - 4, 4) != ".npz")
    file += ".npz";
  auto writer = std::make_shared<io::FileWriter>(std::move(file));
  if (compressed) {
    // Compress each entry with DEFLATE
    if (!writer || !writer->good() || !writer->is_open()) {
      throw std::runtime_error("[savez] Output stream not open");
    }
    std::vector<ZipEntry> entries;
    entries.reserve(arrays.size());
    std::vector<char> scratch;
    std::vector<char> central_dir;
    for (const auto& [name, arr] : arrays) {
      std::string fname = name;
      if (fname.rfind(".npy") == std::string::npos) fname += ".npy";
      MemoryWriter mw;
      array a = arr; a.eval();
      save(std::shared_ptr<io::Writer>(&mw, [](io::Writer*) {}), a);
      const auto& data = mw.data();
      uint32_t crc = crc32_update(0, (const unsigned char*)data.data(), data.size());
      // Compress using zlib (deflate)
      uLongf dest_len = compressBound((uLongf)data.size());
      std::vector<unsigned char> zbuf(dest_len);
      int zrc = compress2(zbuf.data(), &dest_len, (const Bytef*)data.data(), (uLongf)data.size(), Z_BEST_SPEED);
      if (zrc != Z_OK) throw std::runtime_error("[savez] zlib compress failed");
      zbuf.resize(dest_len);
      ZipEntry e;
      e.name = fname;
      e.crc32 = crc;
      e.comp_size = (uint32_t)zbuf.size();
      e.uncomp_size = (uint32_t)data.size();
      e.local_header_offset = (uint32_t)writer->tell();
      write_local_header(*writer, e, /*method=*/8, scratch);
      writer->write((const char*)zbuf.data(), zbuf.size());
      entries.push_back(e);
    }
    uint32_t cd_offset = (uint32_t)writer->tell();
    for (const auto& e : entries) write_central_header(central_dir, e, /*method=*/8);
    if (!central_dir.empty()) writer->write(central_dir.data(), central_dir.size());
    uint32_t cd_size = (uint32_t)central_dir.size();
    write_end_of_central(*writer, (uint16_t)entries.size(), cd_size, cd_offset, scratch);
    return;
  }
  savez(writer, arrays, /*compressed=*/false);
}

} // namespace mlx::core
