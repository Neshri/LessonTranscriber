import sqlite3
from datetime import datetime

# Schema definitions
PROCESSED_FILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'processed'
)
"""

SENT_EMAILS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sent_emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    to_email TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

SENT_SUMMARIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sent_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL UNIQUE,
    summary_name TEXT NOT NULL,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_path TEXT
)
"""

TRANSCRIPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    transcript TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES processed_files (id)
)
"""

def create_tables(conn):
    """Create all tables in the database if they do not exist."""
    conn.execute(PROCESSED_FILES_SCHEMA)
    conn.execute(SENT_EMAILS_SCHEMA)
    conn.execute(SENT_SUMMARIES_SCHEMA)
    conn.execute(TRANSCRIPTS_SCHEMA)
    conn.commit()

def init_db(db_path='lesson_transcriber.db'):
    """Initialize the database by connecting and creating tables."""
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    return conn

# CRUD operations for processed_files
def insert_processed_file(conn, file_path, file_hash, status='processed'):
    """Insert a new processed file record."""
    conn.execute("INSERT OR REPLACE INTO processed_files (file_path, file_hash, status) VALUES (?, ?, ?)", (file_path, file_hash, status))
    conn.commit()
    return conn.lastrowid

def get_all_processed_files(conn):
    """Get all processed files."""
    cursor = conn.execute("SELECT * FROM processed_files")
    return cursor.fetchall()

def get_processed_file_by_id(conn, file_id):
    """Get a processed file by ID."""
    cursor = conn.execute("SELECT * FROM processed_files WHERE id = ?", (file_id,))
    return cursor.fetchone()

def update_processed_file(conn, file_id, file_path=None, file_hash=None, status=None):
    """Update a processed file record."""
    updates = []
    params = []
    if file_path is not None:
        updates.append("file_path = ?")
        params.append(file_path)
    if file_hash is not None:
        updates.append("file_hash = ?")
        params.append(file_hash)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if not updates:
        return
    params.append(file_id)
    conn.execute(f"UPDATE processed_files SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()

def delete_processed_file(conn, file_id):
    """Delete a processed file by ID."""
    conn.execute("DELETE FROM processed_files WHERE id = ?", (file_id,))
    conn.commit()

def get_file_hash_from_db(conn, file_path):
    """Get the stored hash for a file path."""
    cursor = conn.execute("SELECT file_hash FROM processed_files WHERE file_path = ?", (file_path,))
    row = cursor.fetchone()
    return row[0] if row else None

def get_all_processed_files_hashes(conn):
    """Get all processed files as dict of file_path: file_hash."""
    cursor = conn.execute("SELECT file_path, file_hash FROM processed_files")
    rows = cursor.fetchall()
    return {row[0]: row[1] for row in rows}

# CRUD operations for sent_emails
def insert_sent_email(conn, to_email, subject, body):
    """Insert a new sent email record."""
    conn.execute("INSERT INTO sent_emails (to_email, subject, body) VALUES (?, ?, ?)", (to_email, subject, body))
    conn.commit()
    return conn.lastrowid

def get_all_sent_emails(conn):
    """Get all sent emails."""
    cursor = conn.execute("SELECT * FROM sent_emails")
    return cursor.fetchall()

def get_sent_email_by_id(conn, email_id):
    """Get a sent email by ID."""
    cursor = conn.execute("SELECT * FROM sent_emails WHERE id = ?", (email_id,))
    return cursor.fetchone()

def update_sent_email(conn, email_id, to_email=None, subject=None, body=None):
    """Update a sent email record."""
    updates = []
    params = []
    if to_email is not None:
        updates.append("to_email = ?")
        params.append(to_email)
    if subject is not None:
        updates.append("subject = ?")
        params.append(subject)
    if body is not None:
        updates.append("body = ?")
        params.append(body)
    if not updates:
        return
    params.append(email_id)
    conn.execute(f"UPDATE sent_emails SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()

def delete_sent_email(conn, email_id):
    """Delete a sent email by ID."""
    conn.execute("DELETE FROM sent_emails WHERE id = ?", (email_id,))
    conn.commit()

# CRUD operations for sent_summaries
def insert_sent_summary(conn, file_hash, summary_name, file_path=None):
    """Insert a new sent summary record."""
    conn.execute("INSERT OR REPLACE INTO sent_summaries (file_hash, summary_name, file_path) VALUES (?, ?, ?)", (file_hash, summary_name, file_path))
    conn.commit()
    return conn.lastrowid

def get_all_sent_summaries(conn):
    """Get all sent summaries."""
    cursor = conn.execute("SELECT * FROM sent_summaries")
    return cursor.fetchall()

def get_sent_summary_by_hash(conn, file_hash):
    """Get a sent summary by file hash."""
    cursor = conn.execute("SELECT * FROM sent_summaries WHERE file_hash = ?", (file_hash,))
    return cursor.fetchone()

def is_summary_sent(conn, file_hash):
    """Check if summary has been sent by file hash."""
    cursor = conn.execute("SELECT 1 FROM sent_summaries WHERE file_hash = ? LIMIT 1", (file_hash,))
    return cursor.fetchone() is not None

def delete_sent_summary(conn, file_hash):
    """Delete a sent summary by file hash."""
    conn.execute("DELETE FROM sent_summaries WHERE file_hash = ?", (file_hash,))
    conn.commit()

# CRUD operations for transcripts
def insert_transcript(conn, file_id, transcript, summary):
    """Insert a new transcript record."""
    conn.execute("INSERT INTO transcripts (file_id, transcript, summary) VALUES (?, ?, ?)", (file_id, transcript, summary))
    conn.commit()
    return conn.lastrowid

def get_all_transcripts(conn):
    """Get all transcripts."""
    cursor = conn.execute("SELECT * FROM transcripts")
    return cursor.fetchall()

def get_transcript_by_id(conn, transcript_id):
    """Get a transcript by ID."""
    cursor = conn.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
    return cursor.fetchone()

def get_transcripts_by_file_id(conn, file_id):
    """Get transcripts by file ID."""
    cursor = conn.execute("SELECT * FROM transcripts WHERE file_id = ?", (file_id,))
    return cursor.fetchall()

def update_transcript(conn, transcript_id, file_id=None, transcript=None):
    """Update a transcript record."""
    updates = []
    params = []
    if file_id is not None:
        updates.append("file_id = ?")
        params.append(file_id)
    if transcript is not None:
        updates.append("transcript = ?")
        params.append(transcript)
    if not updates:
        return
    params.append(transcript_id)
    conn.execute(f"UPDATE transcripts SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()

def delete_transcript(conn, transcript_id):
    """Delete a transcript by ID."""
    conn.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))
    conn.commit()