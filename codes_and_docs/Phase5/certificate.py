"""
Signed verification certificates.

After an unlearning operation, the server issues a certificate binding
together: the pre-unlearning commitment, the post-unlearning commitment, the
client whose data was removed, the round number, a timestamp, and the
measured MIA success-rate drop -- then signs the whole bundle with an RSA
private key. Any party holding the corresponding public key (patients,
auditors, regulators under DPDP Act Section 12) can verify the certificate
without needing access to the model itself, and without trusting the server's
say-so.
"""

import json
import time
from dataclasses import dataclass, asdict

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature


def generate_server_keypair():
    """Generate a fresh RSA-2048 keypair for the server to sign certificates.
    In a real deployment this would be generated once and the private key
    kept in a hospital's/cluster's HSM or secure key store."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@dataclass
class UnlearningCertificate:
    client_id: int
    round_num: int
    timestamp: float
    pre_commitment: str
    post_commitment: str
    mia_success_before: float
    mia_success_after: float
    signature_hex: str = ""

    def payload_bytes(self) -> bytes:
        """The exact bytes that get signed / verified -- everything except
        the signature itself."""
        payload = {
            "client_id": self.client_id,
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "pre_commitment": self.pre_commitment,
            "post_commitment": self.post_commitment,
            "mia_success_before": self.mia_success_before,
            "mia_success_after": self.mia_success_after,
        }
        return json.dumps(payload, sort_keys=True).encode()

    def to_dict(self):
        return asdict(self)


def issue_certificate(
    private_key,
    client_id: int,
    round_num: int,
    pre_commitment: str,
    post_commitment: str,
    mia_success_before: float,
    mia_success_after: float,
) -> UnlearningCertificate:
    cert = UnlearningCertificate(
        client_id=client_id,
        round_num=round_num,
        timestamp=time.time(),
        pre_commitment=pre_commitment,
        post_commitment=post_commitment,
        mia_success_before=mia_success_before,
        mia_success_after=mia_success_after,
    )
    signature = private_key.sign(
        cert.payload_bytes(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    cert.signature_hex = signature.hex()
    return cert


def verify_certificate(public_key, cert: UnlearningCertificate) -> bool:
    """Returns True iff the certificate's signature is valid for its payload
    under this public key -- i.e. it was genuinely issued by the holder of
    the matching private key and has not been tampered with since."""
    try:
        public_key.verify(
            bytes.fromhex(cert.signature_hex),
            cert.payload_bytes(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False
