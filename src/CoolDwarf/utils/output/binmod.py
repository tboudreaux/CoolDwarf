import hashlib
import pickle
import struct
import numpy as np

from CoolDwarf.star import VoxelSphere

class binmod:
    SOH = bytes([0x01])
    RS = bytes([0x1E])
    ENQ = bytes([0x05])
    STX = bytes([0x02])
    ETX = bytes([0x03])
    EOT = bytes([0x04])

    def build_header(self, star, checksums=False):
        b = bytearray()
        b.extend(self.SOH)
        b.extend(self.RS)

        if checksums:
            b.extend(b'\x00\x00')
            b.extend(self.calculate_md5(star.temperature.to_bytes()))
            b.extend(self.calculate_md5(star.energy.to_bytes()))
            b.extend(self.calculate_md5(star.pressure.to_bytes()))
            b.extend(self.calculate_md5(star.density.to_bytes()))
            b.extend(self.calculate_md5(star.R.to_bytes()))
            b.extend(self.calculate_md5(star.THETA.to_bytes()))
            b.extend(self.calculate_md5(star.PHI.to_bytes()))
        else:
            b.extend(b'\x00')
        pass
        b.extend(self.ENQ)
        b.extend(self.SOH)

        b.extend(star.radialResolution.to_bytes(4, byteorder='little'))
        b.extend(star.azimuthalResolition.to_bytes(4, byteorder='little'))
        b.extend(star.altitudinalResolition.to_bytes(4, byteorder='little'))
        b.extend(struct.pack('d', star.radius))
        b.extend(struct.pack('d', star._mass))
        b.extend(struct.pack('d', star.alpha))
        b.extend(struct.pack('d', star.mindt))
        b.extend(struct.pack('d', star.cfl_factor))
        b.extend(struct.pack('d', star.epsilonH))
        b.extend(struct.pack('d', star._t))
        b.extend(struct.pack('d', star._X))
        b.extend(struct.pack('d', star._Y))
        b.extend(struct.pack('d', star._Z))
        b.extend(struct.pack('d', star._effectiveMolarMass))

        b.extend(self.RS)

        tolDictBytes = pickle.dumps(star._tolerances)
        b.extend(len(tolDictBytes).to_bytes(4, byteorder='little'))
        b.extend(tolDictBytes)

        return b

    def build_body(self, star):
        b = bytearray()
        b.extend(self.STX)

        Rx, Tx, Px = star.temperature.shape
        b.extend(Rx.to_bytes(4, byteorder='little'))
        b.extend(Tx.to_bytes(4, byteorder='little'))
        b.extend(Px.to_bytes(4, byteorder='little'))

        b.extend(self.RS)

        b.extend(star.temperature.flatten().tobytes())
        b.extend(star.density.flatten().tobytes())
        b.extend(star.energy.flatten().tobytes())
        b.extend(star.pressure.flatten().tobytes())
        b.extend(star.mass.flatten().tobytes())

        b.extend(self.ETX)
        b.extend(self.EOT)

        return b

    def save(self, filename: str, star: VoxelSphere, checksums=False):
        header = self.build_header(star, checksums=checksums)
        body = self.build_body(star)
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(body)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            content = f.read()

        offset = 0

        # Read the header
        if content[offset:offset + 1] != self.SOH:
            raise ValueError("Invalid file format: missing SOH at the beginning")
        offset += 1

        if content[offset:offset + 1] != self.RS:
            raise ValueError("Invalid file format: missing RS after SOH")
        offset += 1

        # Skipping checksum part
        checksums = content[offset:offset + 1] == b'\x00\x00'
        offset += 1 + (6 * 16 if checksums else 0)

        if content[offset:offset + 1] != self.ENQ:
            raise ValueError("Invalid file format: missing ENQ")
        offset += 1

        if content[offset:offset + 1] != self.SOH:
            raise ValueError("Invalid file format: missing SOH before resolution")
        offset += 1

        radialResolution = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4
        azimuthalResolition = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4
        altitudinalResolition = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4

        radius = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        _mass = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        alpha = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        mindt = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        cfl_factor = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        epsilonH = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        _t = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        _X = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        _Y = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        _Z = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8
        _effectiveMolarMass = struct.unpack('d', content[offset:offset + 8])[0]
        offset += 8

        if content[offset:offset + 1] != self.RS:
            raise ValueError("Invalid file format: missing RS after header")
        offset += 1

        tolDictLength = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4
        _tolerances = pickle.loads(content[offset:offset + tolDictLength])
        offset += tolDictLength

        # Read the body
        if content[offset:offset + 1] != self.STX:
            raise ValueError("Invalid file format: missing STX at the beginning of the body")
        offset += 1

        Rx = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4
        Tx = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4
        Px = int.from_bytes(content[offset:offset + 4], byteorder='little')
        offset += 4

        if content[offset:offset + 1] != self.RS:
            raise ValueError("Invalid file format: missing RS after dimensions")
        offset += 1

        num_elements = Rx * Tx * Px
        temperature = np.array(struct.unpack(f'{num_elements}d', content[offset:offset + num_elements * 8]))
        offset += num_elements * 8
        density = np.array(struct.unpack(f'{num_elements}d', content[offset:offset + num_elements * 8]))
        offset += num_elements * 8
        energy = np.array(struct.unpack(f'{num_elements}d', content[offset:offset + num_elements * 8]))
        offset += num_elements * 8
        pressure = np.array(struct.unpack(f'{num_elements}d', content[offset:offset + num_elements * 8]))
        offset += num_elements * 8
        mass = np.array(struct.unpack(f'{num_elements}d', content[offset:offset + num_elements * 8]))
        offset += num_elements * 8

        temperature = temperature.reshape((Rx, Tx, Px))
        density = density.reshape((Rx, Tx, Px))
        energy = energy.reshape((Rx, Tx, Px))
        pressure = pressure.reshape((Rx, Tx, Px))
        mass = mass.reshape((Rx, Tx, Px))

        if content[offset:offset + 1] != self.ETX:
            raise ValueError("Invalid file format: missing ETX at the end of the body")
        offset += 1

        if content[offset:offset + 1] != self.EOT:
            raise ValueError("Invalid file format: missing EOT at the end of the file")
        offset += 1

        outDict = {
                "radialResolution": radialResolution,
                "azimuthalResolition": azimuthalResolition,
                "altitudinalResolition": altitudinalResolition,
                "radius": radius,
                "_mass": _mass,
                "alpha": alpha,
                "mindt": mindt,
                "cfl_factor": cfl_factor,
                "epsilonH": epsilonH,
                "_t": _t,
                "_X": _X,
                "_Y": _Y,
                "_Z": _Z,
                "_effectiveMolarMass": _effectiveMolarMass,
                "_tolerances": _tolerances,
                "temperature": temperature,
                "density": density,
                "energy": energy,
                "pressure": pressure,
                "mass": mass
                        }
        return outDict


    @staticmethod
    def calculate_md5(bts):
        md5 = hashlib.md5()
        md5.update(bts)
        return md5.digest()
