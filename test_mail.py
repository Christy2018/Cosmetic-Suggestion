from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

conf = ConnectionConfig(
    MAIL_USERNAME="cosmeticsuggestion@gmail.com",
    MAIL_PASSWORD="pznldqzogesfkvrm",
    MAIL_FROM="cosmeticsuggestion@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,  # ✅ Replace MAIL_TLS
    MAIL_SSL_TLS=False,  # ✅ Replace MAIL_SSL
    USE_CREDENTIALS=True
)

message = MessageSchema(
    subject="Test Email from FastAPI",
    recipients=["christychacko2018@gmail.com"],  # ✅ Change to your real email
    body="This is a test email sent using FastAPI-Mail 🎉",
    subtype="plain"
)

import asyncio

fm = FastMail(conf)

async def send_test_email():
    await fm.send_message(message)
    print("✅ Email sent!")

asyncio.run(send_test_email())
