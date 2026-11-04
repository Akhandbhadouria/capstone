from django.core.mail import send_mail
from django.conf import settings


def send_otp_email(email, otp, purpose='register'):
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

    send_mail(
        subject=subject,
        message=body,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email],
        fail_silently=False,
    )
