import json
import logging
import urllib.request
import urllib.error

from django.conf import settings
from django.core.mail import send_mail

logger = logging.getLogger(__name__)


def _build_message(otp, purpose):
    if purpose == 'register':
        subject = 'ClassAudit AI \u2014 Verify Your Email'
        body = f"""Welcome to ClassAudit AI!

Your email verification OTP is: {otp}

This OTP is valid for 10 minutes.

If you did not create an account, please ignore this email.

\u2014 ClassAudit AI Team"""
    else:
        subject = 'ClassAudit AI \u2014 Your Login OTP'
        body = f"""Hello,

Your login OTP is: {otp}

This OTP is valid for 10 minutes.

If you did not request this, please secure your account immediately.

\u2014 ClassAudit AI Team"""
    return subject, body


def _send_via_brevo(email, subject, body):
    """Send through Brevo's HTTPS API (works on hosts that block SMTP ports)."""
    sender_email = settings.BREVO_SENDER_EMAIL or settings.EMAIL_HOST_USER
    payload = {
        'sender': {'name': 'ClassAudit AI', 'email': sender_email},
        'to': [{'email': email}],
        'subject': subject,
        'textContent': body,
    }
    req = urllib.request.Request(
        'https://api.brevo.com/v3/smtp/email',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'api-key': settings.BREVO_API_KEY,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            logger.info("Brevo accepted OTP email for %s (HTTP %s)", email, resp.status)
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"Brevo API error {e.code}: {detail}") from e


def send_otp_email(email, otp, purpose='register'):
    """
    Sends the OTP email. Raises an exception on failure so callers can react.
    Uses the Brevo HTTPS API if BREVO_API_KEY is set, otherwise Django SMTP.
    """
    subject, body = _build_message(otp, purpose)

    if settings.BREVO_API_KEY:
        _send_via_brevo(email, subject, body)
        return

    sent = send_mail(
        subject=subject,
        message=body,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email],
        fail_silently=False,
    )
    logger.info("SMTP send_mail returned %s for %s", sent, email)
