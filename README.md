# Lesson Transcriber (Whisper + Ollama)

A Python application that transcribes audio lessons using OpenAI's Whisper model and generates concise summaries using Ollama's local LLM capabilities.

## Features

- **Audio Transcription**: Uses Whisper for accurate speech-to-text conversion
- **Intelligent Summarization**: Generates structured summaries using local LLMs via Ollama
- **Multiple Format Support**: Handles MP3, WAV, M4A, FLAC, and OGG files
- **Error Handling**: Robust error handling with detailed logging
- **Command Line Interface**: Simple CLI for easy usage

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** running locally (download from [ollama.com](https://ollama.com/))
3. **Pull a language model** in Ollama (e.g., `ollama pull llama3.2`)

## Installation

1. Clone or download this repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python main.py path/to/your/lesson.mp3
```

### Example Output

The program will:
1. Transcribe the audio file
2. Generate a summary using Ollama
3. Display results on console
4. Save transcript and summary to `output/` directory

==============================================================================
LESSON TRANSCRIPTION SUMMARY
==============================================================================
Audio File: path/to/lesson.mp3
Transcript: output/lesson_transcript.txt
Summary: output/lesson_summary.txt


==============================================================================
TRANSCRIPT:
==============================================================================
[Full transcript text here...]


==============================================================================
SUMMARY:
==============================================================================
[Concise lesson summary here...]

## Configuration

The application uses `config.json` for configuration options:

- **Whisper Model**: Default is "base". Change in `config.json` for different sizes (tiny, small, medium, large)
- **Ollama Model**: Default is "llama3.2". Change in `config.json`
- **Summary Length**: Default max 1000 words. Configurable in `config.json`
- **Output Directory**: Defaults to "output". Can be customized
- **Default Audio Source**: Defaults to "lesson_audio". Configurable via `default_audio_source` in `config.json`
- **Email Recipients**: List of email addresses to receive summaries (requires Azure Graph API setup)

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)

## Troubleshooting

### Ollama Connection Issues

Make sure Ollama is running:
```bash
ollama serve
```

Check available models:
```bash
ollama list
```

### Whisper Installation Issues

If you encounter Torch installation issues, ensure you have:
- Latest pip: `pip install --upgrade pip`
- Xcode command line tools (macOS): `xcode-select --install`
- VS Build tools (Windows) with C++ build tools

### Large Model Performance

For better accuracy with "large" or "medium" models, consider:
- GPU with CUDA support
- Increasing timeout values for long audio files

## Project Structure

```
LessonTranscriber/
├── main.py             # Main application script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── output/             # Generated transcripts and summaries (auto-created)
```

## Contributing

Feel free to:
- Report issues
- Suggest features
- Submit pull requests

## License

This project uses open-source components:
- OpenAI Whisper (MIT License)
- Ollama (MIT License)
- Python standard libraries

# Email Sender (Graph API Integration)

The `email_sender.py` script provides automated email sending of lesson summaries via Microsoft Azure Graph API.

## Prerequisites

1. **Azure Application Registration**: Set up an Azure AD application with the following permissions:
    - `Mail.Send` (Application permission) for sending emails on behalf of users
    - `User.Read` (Delegated permission) for accessing user information

2. **Environment Variables**: Configure the following in your `.env` file:
    ```
    AZURE_CLIENT_ID=your-client-id-here
    AZURE_TENANT_ID=your-tenant-id-here
    AZURE_CLIENT_SECRET=your-client-secret-here
    TARGET_USER_GRAPH_ID=user-principal-name-or-object-id-sending-emails
    EMAIL_RECIPIENTS=recipient1@email.com,recipient2@email.com
    ```

## Features

- **Graph API Integration**: Secure authentication using MSAL and Azure Graph API
- **Summary Detection**: Automatically scans `output/` folder for new `*_summary.txt` files
- **Token Management**: Handles token refresh and expiration automatically
- **Duplicate Prevention**: Tracks sent emails to avoid sending duplicates
- **Batch Processing**: Send multiple summaries at once
- **Error Handling**: Robust error handling with detailed logging
- **Command-Line Interface**: Easy-to-use CLI with multiple options

## Usage

### Basic Usage

Send a specific summary file:
```bash
python email_sender.py --send-summary output/lesson_summary.txt
```

Send all new summaries:
```bash
python email_sender.py --batch-send
```

Interactive mode:
```bash
python email_sender.py
```

Show help:
```bash
python email_sender.py --help
```

### Email Format

Emails are sent as HTML with:
- Subject line: "Lesson Summary: [Lesson Name]"
- Formatted content with lesson title
- Structured summary content
- Automatic sender information

## Configuration Files

### sent_emails.json
Automatically tracks sent emails to prevent duplicates:
```json
{
  "hash1": {
    "summary_name": "Lesson Title",
    "sent_at": "2025-09-10 12:00:00",
    "file_path": "output/lesson_summary.txt"
  }
}
```

## Azure Setup Instructions

1. **Register Azure Application**:
    - Go to Azure Portal > Azure Active Directory > App registrations
    - Create new application registration
    - Note down Client ID and Tenant ID

2. **Configure Permissions**:
    - Add `Mail.Send` and `User.Read` permissions
    - Grant admin consent for application permissions

3. **Create Client Secret**:
    - Go to Certificates & secrets
    - Create new client secret
    - Copy the secret value (you won't see it again!)

4. **Configure Environment**:
    - Copy `.env` template and fill in your values
    - Set `TARGET_USER_GRAPH_ID` to the user account that will send emails

## Troubleshooting

### Common Issues

1. **Token Errors**: Verify Azure application has correct permissions
2. **Authentication Errors**: Check client ID, tenant ID, and client secret
3. **No Recipients**: Ensure `EMAIL_RECIPIENTS` is properly formatted in `.env`
4. **File Not Found**: Verify summary files exist in the specified path

### Error Codes

- `401 Unauthorized`: Token issues - check Azure permissions
- `403 Forbidden`: Insufficient permissions - verify Graph API scopes
- `400 Bad Request`: Malformed request - check email format/validation

## Integration with Main Loop

This script is designed to be used alongside the main transcription loop:

1. Main loop processes audio files → generates summaries in `output/`
2. Email sender monitors `output/` → sends new summaries via Graph API
3. Future integration: Call email sender from transcription completion

---

# Lesson Transcriber (Whisper + Ollama)