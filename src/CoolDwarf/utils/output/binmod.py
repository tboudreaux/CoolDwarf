import hashlib
import pickle
import struct

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

    @staticmethod
    def calculate_md5(bts):
        md5 = hashlib.md5()
        md5.update(bts)
        return md5.digest()
