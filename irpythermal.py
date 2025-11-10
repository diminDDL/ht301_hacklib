#!/usr/bin/env python3

import math
import time
from sys import platform
from time import sleep
from typing import Tuple

import cv2
import numpy as np

SET_CORRECTION = 0 * 4
SET_REFLECTION = 1 * 4
SET_AMB = 2 * 4
SET_HUMIDITY = 3 * 4
SET_EMISSIVITY = 4 * 4
SET_DISTANCE = 5 * 4

ROWS_SPECIAL_DATA = 4


def read_u16(arr_u16, offset):
    """arr: np.uint16"""
    return arr_u16[offset]


def read_f32(arr_u16, offset, step=2):
    """arr: np.uint16"""
    return arr_u16[offset : offset + step].view(np.float32)[0]


def read_u8(arr_u16, offset, step):
    return arr_u16[offset : offset + step].view(np.uint8)


class Camera:
    """Class for reading data from the XTherm/HT301/InfiRay thermal cameras"""

    supported_resolutions = {(240, 180), (256, 192), (384, 288), (640, 512)}
    ZEROC = 273.15
    distance_multiplier = 1.0
    offset_temp_shutter = 0.0
    offset_temp_fpa = 0.0

    range = 120
    cal_00_offset = 390.0
    cal_00_fpamul = 7.05

    correction_coefficient_m = 1
    correction_coefficient_b = 0

    height: int
    width: int
    frame_width: int
    frame_height: int
    frame: np.ndarray
    frame_raw: np.ndarray
    meta: np.ndarray
    device_strings: Tuple[str, str, str, str, str, str]
    userArea: int
    amountPixels: int
    fourLinePara: int
    cap: cv2.VideoCapture
    frame_raw_u16: np.ndarray

    camera_raw = False
    reference_frame = None
    offset_mean = 0.0
    dead_pixels_mask = None

    def __init__(
        self,
        video_dev: cv2.VideoCapture | None = None,
        camera_raw=False,
        fixed_offset: float = 0.0,
    ) -> None:
        if video_dev is None:
            video_dev = self.find_device()
        if not video_dev:
            raise Exception("No video device found!")
        self.cap = video_dev
        self.camera_raw = camera_raw
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - ROWS_SPECIAL_DATA
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        self.userOffset = fixed_offset

        self.fourLinePara = self.width * self.height
        self.init_parameters()
        self.userArea = self.amountPixels + 127

        # Decide whether or not convert data to RGB
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        # using Raw mode 16 bit data
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8004)

        # Wait for the camera to apply the temperature range change
        self.wait_for_range_application()

        # Calibrate the camera
        self.calibrate()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def find_device(cls) -> cv2.VideoCapture:
        """Find a supported thermal camera

        Optionally narrow down the resultion of the camera too look for."""
        for i in range(10):
            try:
                if platform.startswith("linux"):
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                else:
                    cap = cv2.VideoCapture(i)

                cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(
                    f"Found a camera {i} with resolution {int(cap_width)}x{int(cap_height)}"
                )

                if (
                    cap_width,
                    cap_height - ROWS_SPECIAL_DATA,
                ) in cls.supported_resolutions:
                    return cap
            except:
                pass
        raise ValueError(
            f"Cannot find camera with a width of one of {cls.supported_resolutions} that also matches: {cap_width=} and {cap_height=}"
        )

    def bin_to_twos_complement(self, binary: str) -> int:
        if binary[0] == "1":
            return int(binary, 2) - 2 ** len(binary)
        return int(binary, 2)

    def info(self) -> Tuple[dict, np.ndarray]:
        shutTemper = read_u16(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 1
        )  # 257
        if self.camera_raw:
            if shutTemper < 2049:
                floatShutTemper = float(shutTemper)
                corrFactor = 0.625
            else:
                floatShutTemper = float(0xFFF - shutTemper)
                corrFactor = -0.625
            shutterFix = (
                self.bin_to_twos_complement(
                    (
                        bin(
                            read_u16(
                                self.frame_raw_u16,
                                self.fourLinePara + self.amountPixels * 2 + 47,
                            )
                        )[2:].zfill(16)
                    )[:8]
                )
                / 10.0
            )  # 559
            floatShutTemper = (floatShutTemper * corrFactor + 2731.5) / 10.0 + -273.15
            floatShutTemper = floatShutTemper + shutterFix
            # TODO fix this readout for the T2S+ v2
            # The temperature is indeed being red out, but the sensor is located in some weird place,
            # it gets hot super fast and causes the entire image to drift, I couldn't figure out how to deal with that
            # hard coding ~18C (room temp) works pretty good tho...
            # floatShutTemper = 18.0
        else:
            floatShutTemper = shutTemper / 10.0 - self.ZEROC

        coreTemper = read_u16(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 2
        )  # 258
        if self.camera_raw:
            # TODO fix this readout for the T2S+ v2
            # I don't even think the v2 has a separate core and shutter temperature registers...
            # floatCoreTemper = 18.0
            floatCoreTemper = floatShutTemper - shutterFix
            # floatShutTemper = 18.0
        else:
            floatCoreTemper = coreTemper / 10.0 - self.ZEROC

        # print(f"Shutter temperature: {floatShutTemper}°C, Core temperature: {coreTemper / 10.0 - self.ZEROC}°C")

        # TODO check all of these
        cal_00 = float(
            read_u16(self.frame_raw_u16, self.fourLinePara + self.amountPixels)
        )  # 256
        self.cal_01 = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 3
        )  # 259
        cal_02 = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 5
        )  # 261
        cal_03 = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 7
        )  # 263
        cal_04 = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 9
        )  # 265
        cal_05 = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 11
        )  # 267

        cameraSoftVersion: np.ndarray = read_u8(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 24, step=8
        )  # 280-287?
        cameraSoftVersion = cameraSoftVersion.tobytes().decode("ascii").rstrip("\x00")

        # this one is reading out incorrectly?
        sn: np.ndarray = read_u8(
            self.frame_raw_u16, self.fourLinePara + self.amountPixels + 32, step=3
        )  # 288-290?
        sn = sn.tobytes().decode("ascii").rstrip("\x00")

        correction = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.userArea
        )  # 383
        Refltmp = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.userArea + 2
        )  # 385
        Airtmp = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.userArea + 4
        )  # 387
        humi = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.userArea + 6
        )  # 389
        emiss = read_f32(
            self.frame_raw_u16, self.fourLinePara + self.userArea + 8
        )  # 391
        distance = read_u16(
            self.frame_raw_u16, self.fourLinePara + self.userArea + 10
        )  # 393

        fpa_avg = read_u16(self.frame_raw_u16, self.fourLinePara)  # 0
        fpaTmp = read_u16(self.frame_raw_u16, self.fourLinePara + 1)  # 1
        maxx1 = read_u16(self.frame_raw_u16, self.fourLinePara + 2)  # 2
        maxy1 = read_u16(self.frame_raw_u16, self.fourLinePara + 3)  # 3
        self.max_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 4)  # 4
        minx1 = read_u16(self.frame_raw_u16, self.fourLinePara + 5)  # 5
        miny1 = read_u16(self.frame_raw_u16, self.fourLinePara + 6)  # 6
        self.min_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 7)  # 7
        self.avg_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 8)  # 8

        fpatmp_ = 20.0 - (float(fpaTmp) - self.fpa_off) / self.fpa_div

        center_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 12)  # 12
        user_raw00 = read_u16(self.frame_raw_u16, self.fourLinePara + 13)  # 13
        user_raw01 = read_u16(self.frame_raw_u16, self.fourLinePara + 14)  # 14
        user_raw02 = read_u16(self.frame_raw_u16, self.fourLinePara + 15)  # 15

        distance_adjusted = (
            20.0 if distance >= 20.0 else distance
        ) * self.distance_multiplier
        atm = self.atmt(humi, Airtmp, distance_adjusted)
        self.numerator_sub = (1.0 - emiss) * atm * math.pow(Refltmp + self.ZEROC, 4) + (
            1.0 - atm
        ) * math.pow(Airtmp + self.ZEROC, 4)
        self.denominator = emiss * atm

        ts = floatShutTemper + self.offset_temp_shutter
        tfpa = fpatmp_ + self.offset_temp_fpa

        self.cal_a = cal_02 / (self.cal_01 + self.cal_01)
        self.cal_b = cal_02 * cal_02 / (self.cal_01 * self.cal_01 * 4.0)
        self.cal_c = self.cal_01 * math.pow(ts, 2) + ts * cal_02
        self.cal_d = cal_03 * math.pow(tfpa, 2) + cal_04 * tfpa + cal_05

        cal_00_corr = 0

        if self.range == 120:
            cal_00_corr = int(self.cal_00_offset - tfpa * self.cal_00_fpamul)

        table_offset = cal_00 - (cal_00_corr if cal_00_corr > 0 else 0)

        temperatureTable = self.get_temp_table(
            correction, Airtmp, table_offset, distance_adjusted
        )

        temperatureTable = temperatureTable + self.userOffset

        """ build infomation """
        info = {
            "temp_shutter": floatShutTemper,
            "temp_core": floatCoreTemper,
            "cameraSoftVersion": cameraSoftVersion,
            "sn": sn,
            "correction": correction,
            "temp_reflected": Refltmp,
            "temp_air": Airtmp,
            "humidity": humi,
            "emissivity": emiss,
            "distance": distance,
            "fpa_average": fpa_avg,
            "temp_fpa": fpatmp_,
            "temp_max_x": maxx1,
            "temp_max_y": maxy1,
            "temp_max_raw": self.max_raw,
            "temp_max": temperatureTable[self.max_raw],
            "temp_min_x": minx1,
            "temp_min_y": miny1,
            "temp_min_raw": self.min_raw,
            "temp_min": temperatureTable[self.min_raw],
            "temp_average_raw": self.avg_raw,
            "temp_average": temperatureTable[self.avg_raw],
            "temp_center_raw": center_raw,
            "temp_center": temperatureTable[center_raw],
            "temp_user_00": temperatureTable[user_raw00],
            "temp_user_01": temperatureTable[user_raw01],
            "temp_user_02": temperatureTable[user_raw02],
            "Tmin_point": (minx1, miny1),
            "Tmax_point": (maxx1, maxy1),
            "Tcenter_point": (self.width // 2, self.height // 2),
            "Tmin_C": temperatureTable[self.min_raw],
            "Tmax_C": temperatureTable[self.max_raw],
            "Tcenter_C": temperatureTable[center_raw],
        }

        return info, temperatureTable

    # read raw data from cam, seperate visible frame from metadata
    def read(self, raw=False,  max_retries=5, retry_delay=0.5) -> Tuple[bool, np.ndarray]:
        """Returns a raw frame from the camera, retrying if the frame is incomplete."""
        for attempt in range(max_retries):
            ret, frame_raw = self.cap.read()
            if not ret or frame_raw is None:
                time.sleep(retry_delay)
                continue

            frame_data = frame_raw.view(np.uint16).ravel()
            expected_size = self.height * self.width
            if frame_data.size < expected_size:
                # Incomplete frame, wait and retry
                print(
                    f"[warn] Incomplete frame (got {frame_data.size}, expected {expected_size}), retrying {attempt+1}/{max_retries}"
                )
                time.sleep(retry_delay)
                continue

            try:
                frame_visible = frame_data[: self.fourLinePara].copy().reshape(
                    self.height, self.width
                )
                break
            except ValueError:
                print(
                    f"[warn] Reshape failed (array size {frame_data.size}), retrying {attempt+1}/{max_retries}"
                )
                time.sleep(retry_delay)
                continue
        else:
            # If we exit the loop without success
            raise RuntimeError("Failed to read a complete frame from camera")

        self.frame_raw_u16: np.ndarray = frame_raw.view(np.uint16).ravel()

        if raw:
            return ret, frame_visible
        if self.reference_frame is not None:
            frame_float = frame_visible.astype(np.float32)

            corrected_frame = frame_float - self.reference_frame + self.offset_mean

            corrected_frame = np.clip(corrected_frame, 0, 65535)

            if self.dead_pixels_mask is not None:
                inpaint_radius = 3
                corrected_frame = cv2.inpaint(
                    corrected_frame,
                    self.dead_pixels_mask,
                    inpaint_radius,
                    cv2.INPAINT_TELEA,
                )

            frame_visible = corrected_frame.astype(np.uint16)

        return ret, frame_visible

    def get_frame(self) -> np.ndarray:
        """Returns a frame with temperature values"""
        ret, frame = self.read()
        info, lut = self.info()
        lut_frame = lut[frame]
        return lut_frame

    def convert_to_frame(self, frame_raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Converts a raw frame to a frame with temperature values"""
        return lut[frame_raw]

    def set_correction(self, correction: float) -> None:
        self.sendFloatCommand(position=SET_CORRECTION, value=correction)

    def set_reflection(self, reflection: float) -> None:
        self.sendFloatCommand(position=SET_REFLECTION, value=reflection)

    def set_amb(self, amb: float) -> None:
        self.sendFloatCommand(position=SET_AMB, value=amb)

    def set_humidity(self, humidity: float) -> None:
        self.sendFloatCommand(position=SET_HUMIDITY, value=humidity)

    def set_emissivity(self, emiss: float) -> None:
        self.sendFloatCommand(position=SET_EMISSIVITY, value=emiss)

    def set_distance(self, distance: int) -> None:
        self.sendUshortCommand(position=SET_DISTANCE, value=distance)

    """ Control methods"""

    # send command and a float value to camera
    def sendFloatCommand(self, position: int, value: float) -> None:
        # Split float to 4 bytes
        b0, b1, b2, b3 = np.array([value], dtype=np.float32).view(np.uint8)

        positionAndValue0 = (position << 8) | (0x000000FF & b0)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue0):
            print("Control fail {}".format(positionAndValue0))

        positionAndValue1 = ((position + 1) << 8) | (0x000000FF & b1)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue1):
            print("Control fail {}".format(positionAndValue1))

        positionAndValue2 = ((position + 2) << 8) | (0x000000FF & b2)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue2):
            print("Control fail {}".format(positionAndValue2))

        positionAndValue3 = ((position + 3) << 8) | (0x000000FF & b3)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue3):
            print("Control fail {}".format(positionAndValue3))

    # send command and 16 bit value to camera
    def sendUshortCommand(self, position: int, value: int) -> None:
        value0, value1 = np.array([value], dtype=np.uint16).view(np.uint8)
        positionAndValue0 = (position << 8) | (0x000000FF & value0)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue0):
            print("Control fail {}".format(positionAndValue0))

        positionAndValue1 = ((position + 1) << 8) | (0x000000FF & value1)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue1):
            print("Control fail {}".format(positionAndValue1))

    # send command and byte value to camera
    def sendByteCommand(self, position: int, value: int) -> None:
        value0 = np.array([value], dtype=np.uint8)[0]
        psitionAndValue0 = (position << 8) | (0x000000FF & value0)
        self.cap.set(cv2.CAP_PROP_ZOOM, psitionAndValue0)

    # save set parameters
    def save_parameters(self) -> None:
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x80FF)

    # set custom point to measure temperature
    def set_point(self, x: int, y: int, index: int) -> None:
        match index:
            case 0:
                x1 = 0xF000 + x
                y1 = 0xF200 + y
            case 1:
                x1 = 0xF400 + x
                y1 = 0xF600 + y
            case 2:
                x1 = 0xF800 + x
                y1 = 0xFA00 + y
            case _:
                raise ValueError(
                    "Invalid index: {}.\nCan only set 3 custom points to measure temperature at indexes: {}, {}, {}".format(
                        index, 0, 1, 2
                    )
                )

        self.cap.set(cv2.CAP_PROP_ZOOM, x1)
        self.cap.set(cv2.CAP_PROP_ZOOM, y1)

    def calibrate_raw(self, quiet=False) -> None:
        """Camera calibration for cameras that return raw data only"""
        self.reference_frame = None
        self.offset_mean = 0.0
        self.dead_pixels_mask = None
        # uniformity correction
        sleep(0.5)
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)  # close shutter
        sleep(0.3)  # wait for the shutter to close
        self.flush_buffer()
        # by issuing this command faster than once per second, we can keep the shutter closed
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
        ret, frame_visible = self.read(raw=True)

        if ret:
            self.reference_frame = frame_visible.astype(np.float32)
            self.offset_mean = np.mean(self.reference_frame)
        else:
            raise RuntimeError("Failed to capture reference frame")

        # dead pixel correction
        frame_visible_float = frame_visible.astype(np.float32)
        min_val = np.min(frame_visible_float)
        max_val = np.max(frame_visible_float)
        print(f"Min: {min_val}, Max: {max_val}, Avg: {np.mean(frame_visible_float)}")
        threshold_margin = (
            max_val - min_val
        ) * 0.05  # Adjust the margin if not detected correctly
        threshold_lo = min_val + threshold_margin
        threshold_hi = max_val - threshold_margin
        # if there are no dead pixels, we skip the dead pixel correction
        if np.count_nonzero(frame_visible_float < threshold_lo) != 0 or np.count_nonzero(frame_visible_float > threshold_hi) != 0:
            self.dead_pixels_mask = cv2.bitwise_not(cv2.inRange(frame_visible_float, float(threshold_lo), float(threshold_hi))).astype(np.uint8)

        if not quiet:
            print(f"Found {np.count_nonzero(self.dead_pixels_mask)} dead pixels")
            print(f"At: {np.argwhere(self.dead_pixels_mask)}")

    def calibrate(self, quiet=False) -> None:
        """camera calibration"""
        if self.camera_raw:
            self.calibrate_raw(quiet=quiet)
        else:
            self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

    def release(self) -> None:
        """Release cap opencv"""
        self.cap.release()

    def init_parameters(self) -> None:
        """Initalize parameters based on thermal camera resolution"""
        match self.width:
            case 640:
                self.fpa_off = 6867
                self.fpa_div = 33.8
                self.amountPixels = self.width * 3
            case 384:
                self.fpa_off = 7800
                self.fpa_div = 36.0
                self.amountPixels = self.width * 3
            case 256:
                self.fpa_off = 8617
                self.fpa_div = 37.682
                self.amountPixels = self.width
                self.cal_00_offset = 170.0
                self.cal_00_fpamul = 0.0
            case 240:
                self.fpa_off = 7800
                self.fpa_div = 36.0
                self.amountPixels = self.width
            case _:
                raise ValueError(
                    "{} does not match supported device".format(self.width)
                )

    """ Temperature calculation """

    # Water vapor coefficient from humidity and ambient temperature
    def wvc(self, h: float, t_atm: float):
        h1 = 1.5587
        h2 = 0.06939
        h3 = -2.7816e-4
        h4 = 6.8455e-7
        return h * math.exp(
            h1 + h2 * t_atm + h3 * math.pow(t_atm, 2) + h4 * math.pow(t_atm, 3)
        )

    # Transmittance of the atmosphere from humitity, ambient temperature and distance.
    def atmt(self, h: float, t_atm: float, d: float):
        k_atm = 1.9
        nsqd = -math.sqrt(d)
        sqw = math.sqrt(self.wvc(h, t_atm))

        """Athmospheric attenuation without water vapor"""
        a1 = 0.006569
        a2 = 0.01262

        """Attenuation for water vapor."""
        b1 = -0.002276
        b2 = -0.00667
        return k_atm * math.exp(nsqd * (a1 + b1 * sqw)) + (1.0 - k_atm) * math.exp(
            nsqd * (a2 + b2 * sqw)
        )

    # calculate temperature table
    # for each 16 bit value from frame data will return correspond temperture value
    def get_temp_table(self, correction, Airtmp, table_offset, distance_adjusted):
        """x: uint16"""
        n = np.sqrt(
            np.abs(
                (
                    (np.arange(16384, dtype=np.float32) - table_offset) * self.cal_d
                    + self.cal_c
                )
                / self.cal_01
                + self.cal_b
            )
        )
        n[np.isnan(n)] = 0.0
        wtot = np.power(n - self.cal_a + self.ZEROC, 4)
        ttot = (
            np.power((wtot - self.numerator_sub) / self.denominator, 0.25) - self.ZEROC
        )
        # TODO, ttot is being directly read out according to the decomp, how is it calculated here exactly? (fVar7) offset ~269
        temperatureTable = (
            ttot
            + (distance_adjusted * 0.85 - 1.125) * (ttot - Airtmp) / 100.0
            + correction
        )
        return (
            self.correction_coefficient_m * temperatureTable
            + self.correction_coefficient_b
        )

    def temperature_range_normal(self):
        """Switch camera to the normal temperature range (-20°C to 120°C)"""
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8020)
        self.correction_coefficient_m = 1
        self.correction_coefficient_b = 0

    def temperature_range_high(self):
        """Switch camera to the high temperature range (-20°C to 450°C)"""
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8021)
        if self.camera_raw:
            # TODO verify these
            self.correction_coefficient_m = 0.1
            self.correction_coefficient_b = 0
            return
        self.correction_coefficient_m = 1.17
        self.correction_coefficient_b = -40.9

    def wait_for_range_application(self, timeout=20):
        """Wait for the camera to apply the temperature range change"""
        print("Waiting for camera to stabilize...")
        start_time = time.time()
        done = False
        while time.time() - start_time < timeout:
            ret, frame_visible = self.read()
            if ret and np.std(frame_visible) > 0:
                done = True
                break
            time.sleep(0.1)

        if self.camera_raw:
            # Now we keep the shutter closed and wait for the camera to stabilize,
            # we do this by running the calibration, waiting a bit and checking the average
            # of all the pixels, when the change gets below a certain threshold we can consider
            # the camera to be stable.
            # Throughout this routine we keep the shutter closed.
            lowest = 1000
            margin = 0.1
            min_val = 0.01
            while time.time() - start_time < timeout:
                self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
                self.calibrate(quiet=True)
                ret, frame_visible = self.read()
                if ret:
                    # calculate how uniform the frame is
                    std = np.std(frame_visible)
                    self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
                    sleep(0.1)
                    if std > min_val and lowest - std < margin:
                        print(f"Camera is stable with std: {std}")
                        return True

                    if std < lowest and std > min_val:
                        lowest = std

        elif done:
            print("Camera is stable")
            return True

        return False

    def flush_buffer(self, num_reads=16):
        for i in range(num_reads):
            ret, frame_visible = self.read(raw=True)


class MockVidoCapture:
    def set(self, propId, value):
        setattr(self, str(propId), value)

    def get(self, propId):
        return getattr(self, str(propId))


class CameraEmulator(Camera):
    def __init__(self, filename):
        frame_raw_u16 = np.load(filename, allow_pickle=True)
        self.cap = MockVidoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_raw_u16.shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_raw_u16.shape[1])
        self.frame_raw_u16 = frame_raw_u16.ravel()
        super().__init__(video_dev=self.cap)

    def read(self) -> Tuple[bool, np.ndarray]:
        ret = True
        frame_visible = (
            self.frame_raw_u16[: self.fourLinePara]
            .copy()
            .reshape(self.height, self.width)
        )
        return ret, frame_visible
