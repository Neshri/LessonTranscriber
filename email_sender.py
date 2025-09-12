#!/usr/bin/env python3
"""
Lesson Summary Email Sender - Sends lesson summaries via Azure Graph API
"""

import sys
import logging
import os
import requests
import json
import jwt
import time
import hashlib
from pathlib import Path
import markdown
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, environment variables must be set manually")

try:
    import msal
except ImportError:
    msal = None


# Required environment variables
REQUIRED_ENV_VARS = [
    'AZURE_CLIENT_ID',
    'AZURE_TENANT_ID',
    'AZURE_CLIENT_SECRET',
    'TARGET_USER_GRAPH_ID',
]

@dataclass
class EmailConfig:
    """Configuration for email sending"""
    client_id: str
    tenant_id: str
    client_secret: str
    target_user_graph_id: str
    recipients: List[str]
    graph_api_endpoint: str = "https://graph.microsoft.com/v1.0"

class AzureToken:
    """Azure token management class"""

    def __init__(self, client_id: str, tenant_id: str, client_secret: str):
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.client_secret = client_secret
        self.token = None
        self.token_expires_at = None

    def get_token(self) -> str:
        """Get a valid access token, refreshing if necessary"""
        if not self.is_token_valid():
            self._refresh_token()

        if not self.token:
            raise Exception("Failed to acquire access token")

        return self.token

    def is_token_valid(self) -> bool:
        """Check if current token is still valid"""
        if not self.token or not self.token_expires_at:
            return False

        # Refresh token if it's about to expire (5 minutes buffer)
        return time.time() < (self.token_expires_at - 300)

    def _refresh_token(self):
        """Acquire a new access token"""
        if not msal:
            raise ImportError("msal package is required for Azure authentication. Install with: pip install msal")

        # MSAL confidential client application
        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}"
        )

        logger.info("Acquiring Azure token...")

        # Get token for Microsoft Graph scope
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])

        if "access_token" in result:
            self.token = result["access_token"]
            # Decode token to get expiration time
            decoded_token = jwt.decode(self.token, options={"verify_signature": False})
            self.token_expires_at = decoded_token.get("exp", time.time() + 3600)  # Default to 1 hour if not found
            logger.info("Token acquired successfully")
        else:
            error_message = result.get("error_description", "Unknown error while acquiring token")
            raise Exception(f"Failed to acquire token: {error_message}")

def load_config_from_env(recipients: Optional[List[str]] = None) -> EmailConfig:
    """Load email configuration from environment variables or provided parameters"""
    missing_vars = []
    config = {}

    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            config[var.lower()] = value

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Get recipients - use provided parameter or fall back to environment variable
    if recipients is None:
        recipients = []
        recipients_str = os.getenv('EMAIL_RECIPIENTS', '')
        if recipients_str:
            recipients = [email.strip() for email in recipients_str.split(',') if email.strip()]

    return EmailConfig(
        client_id=config['azure_client_id'],
        tenant_id=config['azure_tenant_id'],
        client_secret=config['azure_client_secret'],
        target_user_graph_id=config['target_user_graph_id'],
        recipients=recipients
    )

def make_graph_api_call(url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, token: Optional[str] = None, config: Optional[EmailConfig] = None) -> Dict[str, Any]:
    """
    Make an authenticated call to Microsoft Graph API

    Args:
        url: Graph API endpoint URL
        method: HTTP method (GET, POST, etc.)
        data: Request body data
        token: Access token (will create one if not provided)
        config: Email configuration (will load from env if not provided)

    Returns:
        Dict containing the response
    """
    if not config:
        config = load_config_from_env()

    if not token:
        azure_token = AzureToken(config.client_id, config.tenant_id, config.client_secret)
        token = azure_token.get_token()

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    try:
        logger.info(f"Making {method} request to {url}")

        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, data=json.dumps(data) if data else None, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        logger.info(f"Request completed with status: {response.status_code}")

        if response.status_code == 401:
            logger.warning("Token expired, will need to refresh on next request")
        elif response.status_code >= 400:
            logger.error(f"Graph API error: {response.status_code} - {response.text}")

        response.raise_for_status()

        # For non-empty responses, return the JSON
        if response.content:
            return response.json()

        return {}

    except requests.exceptions.RequestException as e:
        logger.error(f"Graph API request failed: {e}")
        raise Exception(f"Graph API request failed: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Graph API response: {e}")
        raise Exception(f"Invalid JSON response from Graph API: {e}")

def graph_send_email(subject: str, body: str, recipients: List[str], config: EmailConfig) -> bool:
    """
    Send email using Microsoft Graph API

    Args:
        subject: Email subject line
        body: Email body content (HTML format recommended)
        recipients: List of recipient email addresses
        config: Email configuration

    Returns:
        bool: True if email sent successfully
    """
    url = f"{config.graph_api_endpoint}/users/{config.target_user_graph_id}/sendMail"

    # Build email message
    email_data = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body
            },
            "toRecipients": [
                {
                    "emailAddress": {
                        "address": recipient
                    }
                } for recipient in recipients
            ]
        }
    }

    try:
        response = make_graph_api_call(url, method="POST", data=email_data, config=config)
        logger.info(f"Email sent successfully to {len(recipients)} recipients")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

class EmailSender:
    """Main email sender class for lesson summaries"""

    def __init__(self, config: Optional[EmailConfig] = None, recipients: Optional[List[str]] = None):
        """
        Initialize the EmailSender.

        Args:
            config: Optional EmailConfig object, will load from env if not provided
            recipients: Optional list of recipient emails, overrides env var if provided
        """
        if config is None:
            config = load_config_from_env(recipients=recipients)
        self.config = config
        self.sent_emails_file = Path("sent_emails.json")
        self.sent_emails = self._load_sent_emails()

    def _load_sent_emails(self) -> Dict[str, str]:
        """Load tracking of sent emails"""
        if self.sent_emails_file.exists():
            try:
                with open(self.sent_emails_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Failed to load sent emails tracking, starting fresh")
                return {}
        return {}

    def _save_sent_emails(self):
        """Save tracking of sent emails"""
        try:
            with open(self.sent_emails_file, 'w', encoding='utf-8') as f:
                json.dump(self.sent_emails, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save sent emails tracking: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate SHA256 hash of file content"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except (OSError, IOError):
            return None

    def _is_summary_sent(self, summary_path: Path) -> bool:
        """Check if summary has already been sent"""
        file_hash = self._get_file_hash(summary_path)
        if file_hash is None:
            return False
        return file_hash in self.sent_emails

    def _mark_summary_sent(self, summary_path: Path, summary_name: str):
        """Mark summary as sent"""
        file_hash = self._get_file_hash(summary_path)
        if file_hash:
            self.sent_emails[file_hash] = {
                'summary_name': summary_name,
                'sent_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'file_path': str(summary_path)
            }
            self._save_sent_emails()

    def send_summary_email(self, summary_path: Path) -> bool:
        """
        Send a single lesson summary via email

        Args:
            summary_path: Path to the summary .txt file

        Returns:
            bool: True if sent successfully
        """
        if not summary_path.exists():
            logger.error(f"Summary file not found: {summary_path}")
            return False

        if self._is_summary_sent(summary_path):
            logger.info(f"Summary already sent: {summary_path}")
            return False

        # Log configuration details for debugging
        logger.info(f"Email configuration: TARGET_USER_GRAPH_ID={self.config.target_user_graph_id}")
        logger.info(f"Recipients: {self.config.recipients}")

        try:
            # Read summary content
            summary_content = summary_path.read_text(encoding='utf-8')
            summary_name = summary_path.stem.replace('_summary', '').replace('_', ' ').title()

            # Convert markdown to HTML
            html_content = markdown.markdown(summary_content)

            # Create email subject and body
            subject = f"Lesson Summary: {summary_name}"

            # Format body as HTML
            body = f"""
            <html>
            <body>
                <h1>Lesson Summary: {summary_name}</h1>
                <h2>Summary Content</h2>
                <div>
                {html_content}
                </div>
                <hr>
                <p><em>This summary was automatically generated from the lesson transcription.</em></p>
            </body>
            </html>
            """

            # Send email
            success = graph_send_email(
                subject=subject,
                body=body,
                recipients=self.config.recipients,
                config=self.config
            )

            if success:
                self._mark_summary_sent(summary_path, summary_name)
                logger.info(f"Lesson summary emailed: {summary_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to send summary email: {e}")

        return False

    def send_all_new_summaries(self, output_dir: str = "output") -> int:
        """
        Send all new summary files in the output directory

        Args:
            output_dir: Path to the output directory

        Returns:
            int: Number of summaries sent
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.warning(f"Output directory not found: {output_path}")
            return 0

        # Find all summary files
        summary_files = list(output_path.glob("*_summary.txt"))

        if not summary_files:
            logger.info("No summary files found")
            return 0

        sent_count = 0
        for summary_file in summary_files:
            try:
                if self.send_summary_email(summary_file):
                    sent_count += 1
            except Exception as e:
                logger.error(f"Failed to process {summary_file}: {e}")
                continue

        logger.info(f"Sent {sent_count} new lesson summaries")
        return sent_count

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Send lesson summaries via Microsoft Graph API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables required:
  AZURE_CLIENT_ID     - Azure application client ID
  AZURE_TENANT_ID     - Azure tenant ID
  AZURE_CLIENT_SECRET - Azure application client secret
  TARGET_USER_GRAPH_ID - ID of the user/account sending emails
  EMAIL_RECIPIENTS    - Comma-separated list of recipient emails

Examples:
  python email_sender.py                                # Interactive mode
  python email_sender.py --batch-send                   # Send all new summaries
  python email_sender.py --send-summary output/lesson_summary.txt
        """
    )

    parser.add_argument('--send-summary', type=str,
                        help='Path to specific summary file to send')
    parser.add_argument('--batch-send', action='store_true',
                        help='Send all new summaries in output directory')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory containing summary files (default: output/)')

    args = parser.parse_args()

    try:
        # Initialize email sender
        email_sender = EmailSender()

        if args.send_summary:
            # Send specific summary
            summary_path = Path(args.send_summary)
            success = email_sender.send_summary_email(summary_path)
            if success:
                print(f"✅ Summary sent: {summary_path}")
            else:
                print("❌ Failed to send summary")

        elif args.batch_send:
            # Send all new summaries
            sent_count = email_sender.send_all_new_summaries(args.output_dir)
            print(f"✅ Sent {sent_count} new lesson summaries")

        else:
            # Interactive mode - ask user what to do
            print("Lesson Summary Email Sender")
            print("==========================")
            print("1. Send a specific summary file")
            print("2. Send all new summaries")
            print("3. Show sent summaries")

            choice = input("Choose an option (1-3): ").strip()

            if choice == '1':
                file_path = input("Enter path to summary file: ").strip()
                if file_path:
                    success = email_sender.send_summary_email(Path(file_path))
                    if success:
                        print("✅ Summary sent successfully")
                    else:
                        print("❌ Failed to send summary")

            elif choice == '2':
                sent_count = email_sender.send_all_new_summaries(args.output_dir)
                print(f"✅ Sent {sent_count} new lesson summaries")

            elif choice == '3':
                # Show sent summaries
                if email_sender.sent_emails:
                    print("\nSent Summaries:")
                    print("===============")
                    for hash_val, info in email_sender.sent_emails.items():
                        print(f"- {info['summary_name']} (sent {info['sent_at']})")
                else:
                    print("No sent summaries found")
            else:
                print("Invalid choice")

    except Exception as e:
        logger.error(f"Email sender failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()