
import struct
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
import pandas as pd
from io import BytesIO
from typing import Dict, List
from typing import override

from shared_codes.utils.config_utils import get_s3_config


def _open_or_create(name: str, size: int, create: bool, force_create: bool, track: int):
    """
    Open an existing shared-memory block, or create (and optionally re-create)
    it.  Returns the multiprocessing.shared_memory.SharedMemory object.
    """
    if create:
        try:
            return shared_memory.SharedMemory(create=True, size=size, name=name, track=track)
        except FileExistsError:
            if force_create:
                shared_memory.SharedMemory(name=name).unlink()
                return shared_memory.SharedMemory(create=True, size=size, name=name, track=track)

    return shared_memory.SharedMemory(name=name, track=track)


class ShmAbstract(ABC):
    @abstractmethod
    def __init__(self, name: str, create: bool, force_create: bool, total_size: int, logger_name: str, config_object: object, track: int):
        pass

    @abstractmethod
    def _write_metadata(self):
        pass

    @abstractmethod
    def get_metadata(self) -> tuple:
        """
        Return the metadata stored in the shared memory.
        ---------
        Returns:
            tuple: (METADATA_FORMAT, METADATA_SIZE)
        """
        pass

    @abstractmethod
    def _read_metadata(self) -> tuple:
        pass

    @abstractmethod
    def meta_data_check(self):
        pass

class BaseShm(ShmAbstract):
    def __init__(self, name: str, create: bool, force_create: bool, total_size: int, logger_name: str, config_object: object, track: int):
        super().__init__(name, create, force_create, total_size, logger_name, config_object, track)

        self.config_object = config_object
        self.logger_name = logger_name

        ##
        self.shm = _open_or_create(name=name, size=total_size, create=create, force_create=force_create, track=track)
        self.buf = self.shm.buf

    def _close(self):
        self.shm.close()
    
    def _unlink(self):
        self.shm.unlink()



def get_shm_symbol_df():
    get_s3_config.cache_clear()
    data_ = get_s3_config("shm_tradefi_symbols_indx")[0]
    df = pd.read_csv(BytesIO(data_))
    return df

####
class DbnShmSharedFunctions:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_sym_indx_local_cache(self):
        ##
        msg_ = "Initializing symbol index local cache"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
        
        indx_df = get_shm_symbol_df()

        self.symbol_shmIndex_map = {}
        for tup in indx_df.itertuples():
            self.symbol_shmIndex_map[tup.Symbol] = tup.SharedMemIndex

            if tup.SharedMemIndex >= self.num_instruments:
                msg_ = f"Shared memory index {tup.SharedMemIndex} for symbol {tup.Symbol} is greater than num_instruments {self.num_instruments}"
                self.config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)
                raise ValueError(msg_)
        
        # local cache
        self.LOCAL_ROWS: Dict[int, List[int]] = {}
        self.LOCAL_VERSION: Dict[int, int] = {}

        ## To be put in function signal.USR1 handler
        for symbol,indx in self.symbol_shmIndex_map.items():
            row_data = [0]*self.num_fields_per_row
            row_data[self.OFF_INSTR_ID] = indx

            if symbol not in self.LOCAL_ROWS:
                self.LOCAL_ROWS[symbol] = row_data
                self.LOCAL_VERSION[symbol] = 0

    ##
    def _row_offset(self, symbol: str) -> int:
        """Return the byte offset where this instrument's row is stored."""
        # metadata is at offset 0..METADATA_SIZE-1
        # so we start instrument rows at offset = METADATA_SIZE
        row_index = self.symbol_shmIndex_map[symbol]
        return self.METADATA_SIZE + row_index * self.row_size     

    ##
    def _write_instrument_row(self, row_data: List[int], symbol: str):
        """Write local cache of 'inst_id' to shared memory with new version."""
        # row_data = self.LOCAL_ROWS[symbol]
        
        # self.LOCAL_VERSION[symbol] += 1
        # ver = self.LOCAL_VERSION[symbol]
        
        # row_data[self.OFF_VERSION_START] = ver
        # row_data[self.OFF_VERSION_END]   = ver
        
        offset = self._row_offset(symbol)
        struct.pack_into(self.row_format, self.buf, offset, *row_data)
        


class BidAskShmBase(ShmAbstract, DbnShmSharedFunctions):
    METADATA_FORMAT = "q q"  # two 64-bit integers # num_instruments, num_levels
    METADATA_SIZE = struct.calcsize(METADATA_FORMAT)  # 16 bytes
    
    def __init__(self, name: str, create: bool, num_instruments: int, num_levels: int, force_create: bool, config_object: object, logger_name: str, track: int):
        self.num_instruments = num_instruments
        self.num_levels = num_levels
        self.config_object = config_object
        self.logger_name = logger_name

        # ---- 1) Calculate row layout for instruments (like before) ----
        # row contains:
        #   instrument_id (q)
        #   version_start (q)
        #   3*num_levels for bids (each level => (price, size, count))
        #   3*num_levels for asks
        #   ltp_price (q), ltp_size (q)
        #   row_timestamp (q)
        #   version_end (q)
        #
        # => total fields = 2 + 3*num_levels + 3*num_levels + 2 + 1 + 1
        #                 = 4*num_levels + 6
        self.num_ba_fields_per_level = 3
        self.num_fields_per_row = 2 * self.num_ba_fields_per_level * num_levels + 6

        # Each field is an 8-byte 'q'
        self.row_format = "q" * self.num_fields_per_row
        self.row_size = struct.calcsize(self.row_format)
        
        # ---- 2) Calculate total shared memory size ----
        #  [metadata (16 bytes)] + [num_instruments * row_size]
        self.total_size = self.METADATA_SIZE + self.row_size * num_instruments

        msg_ = f"Name: {name}, num_instruments: {num_instruments}, num_levels: {num_levels}, create: {create}, force_create: {force_create}\n"
        msg_ += f"Total size: {self.total_size}, row_size: {self.row_size}, num_fields_per_row: {self.num_fields_per_row}\n"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)
        
        # ---- 3) Create or attach to shared memory ----
        super().__init__(name=name, create=create, force_create=force_create, total_size=self.total_size, logger_name=logger_name, config_object=config_object, track=track)

        ##
        self._IDX_FMT  = "q"                     # 8-byte signed long long
        self._IDX_SIZE = struct.calcsize(self._IDX_FMT)        

        ##
        # def _open_or_create(name: str, size: int, create: bool, force_create: bool, track: int):
        shm0_name = f"{name}_0"
        shm1_name = f"{name}_1"
        shmI_name = f"{name}_idx"

        msg_ = f"Opening/Creating shared memory: {shm0_name}, {shm1_name}, {shmI_name}"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        self.shm0 = _open_or_create(name=shm0_name, size=self.total_size, create=create, force_create=force_create, track=track)
        self.shm1 = _open_or_create(name=shm1_name, size=self.total_size, create=create, force_create=force_create, track=track)
        self.shmI = _open_or_create(name=shmI_name, size=self._IDX_SIZE, create=create, force_create=force_create, track=track)

        ##
        self.buf0 = self.shm0.buf
        self.buf1 = self.shm1.buf
        self.bufI = self.shmI.buf

        # ---- 4) If create=True, write metadata. If not, read & verify. ----
        if create:
            struct.pack_into(self._IDX_FMT, self.bufI, 0, 0)
            self._write_metadata(self.shm0.buf, num_instruments, num_levels)
            self._write_metadata(self.shm1.buf, num_instruments, num_levels)   
        else:
            self._verify_metadata(self.buf0)
            self._verify_metadata(self.buf1)         

        ##
        # Offsets within each row
        self.OFF_INSTR_ID      = 0
        self.OFF_VERSION_START = 1
        self.OFF_BIDS          = 2
        self.OFF_ASKS          = self.OFF_BIDS + self.num_ba_fields_per_level*num_levels
        self.OFF_LTP_PRICE     = self.OFF_ASKS + self.num_ba_fields_per_level*num_levels
        self.OFF_LTP_SIZE      = self.OFF_LTP_PRICE + 1
        self.OFF_ROW_TIMESTAMP = self.OFF_LTP_SIZE + 1
        self.OFF_VERSION_END   = self.OFF_ROW_TIMESTAMP + 1

    ############################################################################
    # Metadata: num_instruments, num_levels
    ############################################################################

    # ─── index helpers ───────────────────────────────────────────────────────
    def _front_idx(self) -> int:
        return struct.unpack_from(self._IDX_FMT, self.bufI, 0)[0]

    def _toggle_idx(self, idx: int) -> int:
        """Return 0→1 or 1→0 (single-writer, two-buffer scheme)."""
        return idx ^ 1           # or int(not idx)  – your choice

    def _set_front_idx(self, idx: int):
        struct.pack_into(self._IDX_FMT, self.bufI, 0, idx)

    def _front_buf(self) -> memoryview:
        return self.buf0 if self._front_idx() == 0 else self.buf1

    def _back_buf(self) -> memoryview:
        return self.buf1 if self._front_idx() == 0 else self.buf0    

    def get_metadata(self) -> tuple:
        return self.METADATA_FORMAT, self.METADATA_SIZE

    ##
    @override
    def _write_instrument_row(self, row_data: List[int], symbol: str):
        """Write local cache of 'inst_id' to shared memory with new version."""
        back = self._back_buf()

        offset = self._row_offset(symbol)
        struct.pack_into(self.row_format, back, offset, *row_data)
        
        ## Set front index value
        new_front_idx = self._toggle_idx(self._front_idx())
        self._set_front_idx(idx=new_front_idx)

    def _write_metadata(self, buf: memoryview, num_instruments: int, num_levels: int):
        """Write the two 64-bit metadata fields at the start of the buffer."""
        struct.pack_into(self.METADATA_FORMAT, buf, 0, num_instruments, num_levels)
    
    def _read_metadata(self, buf: memoryview) -> tuple:
        return struct.unpack_from(self.METADATA_FORMAT, buf, 0)
    
    def _verify_metadata(self, buf: memoryview):
        n, l = self._read_metadata(buf)
        if (n, l) != (self.num_instruments, self.num_levels):
            msg_ = f"Metadata mismatch! Shared memory has (num_instruments={n}) but Config has ({self.num_instruments})."
            self.config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)
            raise ValueError(msg_)
        

    def meta_data_check(self):
        self._verify_metadata(self.buf0)
        self._verify_metadata(self.buf1)

    # ─── tidy up ─────────────────────────────────────────────────────────────
    def _close(self):
        for shm in (self.shm0, self.shm1, self.shmI):
            shm.close()

    def _unlink(self):
        for shm in (self.shm0, self.shm1, self.shmI):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass




class InstrumentStatusShmBase(BaseShm, DbnShmSharedFunctions):
    METADATA_FORMAT = "i" # just a single integer for now # num_instruments
    METADATA_SIZE = struct.calcsize(METADATA_FORMAT)  # 4 bytes

    def __init__(self, name: str, create: bool, num_instruments: int, force_create: bool, config_object: object, logger_name: str, track: int):
        self.num_instruments = num_instruments
        self.config_object = config_object
        self.logger_name = logger_name

        # We have 500 instruments, each row is 24 bytes:
        #   (instrument_id [4], version_start [4], halted_flag [4], timestamp_ns [8], version_end [4])
        # => "i i i q i"
        self.row_format = "i i i q i"
        self.row_size = struct.calcsize(self.row_format)
        self.total_size = self.METADATA_SIZE + self.row_size * num_instruments

        # row contains:
        #   instrument_id (i)
        #   version_start (i)
        #   halted_flag (i)
        #   timestamp_ns (q)
        #   version_end (i)
        # 
        #  => total fields = 5
        self.num_fields_per_row = 5

        msg_ = f"Name: {name}, num_instruments: {num_instruments}, create: {create}, force_create: {force_create}\n"
        msg_ += f"Total size: {self.total_size}, row_size: {self.row_size}\n"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        super().__init__(name=name, create=create, force_create=force_create, total_size=self.total_size, logger_name=logger_name, config_object=config_object, track=track)

        # ---- 4) If create=True, write metadata. If not, read & verify. ----
        if create:
            self._write_metadata(num_instruments)

        ##
        self.OFF_INSTR_ID      = 0
        self.OFF_VERSION_START = 1
        self.OFF_HALTED_FLAG   = 2
        self.OFF_TIMESTAMP_NS  = 3
        self.OFF_VERSION_END   = 4

    def get_metadata(self) -> tuple:
        return self.METADATA_FORMAT, self.METADATA_SIZE

    def _get_row_offset(self, symbol: str) -> int:
        row_index = self.symbol_shmIndex_map[symbol]
        return self.METADATA_SIZE + row_index * self.row_size
    
    def _write_metadata(self, num_instruments: int):
        struct.pack_into(self.METADATA_FORMAT, self.buf, 0, num_instruments)
    
    def _read_metadata(self) -> int:
        return struct.unpack_from(self.METADATA_FORMAT, self.buf, 0)[0]

    def _row_offset(self, symbol: str) -> int:
        row_index = self.symbol_shmIndex_map[symbol]
        return self.METADATA_SIZE + row_index * self.row_size

    def meta_data_check(self):
        msg_ = "Doing metadata check"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        stored_instruments = self._read_metadata()
        if stored_instruments != self.num_instruments:
            msg_ = f"Metadata mismatch! Shared memory has (num_instruments={stored_instruments}) but Config has ({self.num_instruments})."
            self.config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)

            raise ValueError(msg_)


##
class InstrumentAtrSnapshotShmBase(BaseShm, DbnShmSharedFunctions):
    METADATA_FORMAT = "i" # just a single integer for now # num_instruments
    METADATA_SIZE = struct.calcsize(METADATA_FORMAT)  # 4 bytes

    def __init__(self, name: str, create: bool, num_instruments: int, force_create: bool, config_object: object, logger_name: str, track: int):
        self.num_instruments = num_instruments
        self.config_object = config_object
        self.logger_name = logger_name

        # We have 500 instruments, each row is 24 bytes:
        #   (instrument_id [4], version_start [4], atr [4], timestamp_ns [8], version_end [4])
        # => "i i i q i"
        self.row_format = "i i i q i"
        self.row_size = struct.calcsize(self.row_format)
        self.total_size = self.METADATA_SIZE + self.row_size * num_instruments

        # row contains:
        #   instrument_id (i)
        #   version_start (i)
        #   atr (i)
        #   timestamp_ns (q)
        #   version_end (i)
        # 
        #  => total fields = 5
        self.num_fields_per_row = 5

        msg_ = f"Name: {name}, num_instruments: {num_instruments}, create: {create}, force_create: {force_create}\n"
        msg_ += f"Total size: {self.total_size}, row_size: {self.row_size}\n"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        super().__init__(name=name, create=create, force_create=force_create, total_size=self.total_size, logger_name=logger_name, config_object=config_object, track=track)

        # ---- 4) If create=True, write metadata. If not, read & verify. ----
        if create:
            self._write_metadata(num_instruments)

        ##
        self.OFF_INSTR_ID      = 0
        self.OFF_VERSION_START = 1
        self.OFF_ATR_INT   = 2
        self.OFF_TIMESTAMP_NS  = 3
        self.OFF_VERSION_END   = 4

    def get_metadata(self) -> tuple:
        return self.METADATA_FORMAT, self.METADATA_SIZE

    def _get_row_offset(self, symbol: str) -> int:
        row_index = self.symbol_shmIndex_map[symbol]
        return self.METADATA_SIZE + row_index * self.row_size
    
    def _write_metadata(self, num_instruments: int):
        struct.pack_into(self.METADATA_FORMAT, self.buf, 0, num_instruments)
    
    def _read_metadata(self) -> int:
        return struct.unpack_from(self.METADATA_FORMAT, self.buf, 0)[0]

    def _row_offset(self, symbol: str) -> int:
        row_index = self.symbol_shmIndex_map[symbol]
        return self.METADATA_SIZE + row_index * self.row_size

    def meta_data_check(self):
        msg_ = "Doing metadata check"
        self.config_object.masterlog.get_logger(self.logger_name, "info")(msg_)

        stored_instruments = self._read_metadata()
        if stored_instruments != self.num_instruments:
            msg_ = f"Metadata mismatch! Shared memory has (num_instruments={stored_instruments}) but Config has ({self.num_instruments})."
            self.config_object.masterlog.get_logger(self.logger_name, "critical")(msg_)

            raise ValueError(msg_)

