import os
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class EmailChunk:
    """Represents a chunk of an email document"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class EmailProcessor:
    """Handles loading and chunking of email documents"""
    
    def __init__(self, emails_dir: str = "emails"):
        self.emails_dir = emails_dir
        self.chunk_size = 500  # characters per chunk
        self.chunk_overlap = 100  # characters overlap between chunks
    
    def load_emails(self) -> List[Dict[str, Any]]:
        """Load all email files from the directory"""
        emails = []
        
        if not os.path.exists(self.emails_dir):
            raise FileNotFoundError(f"Emails directory '{self.emails_dir}' not found")
        
        for filename in sorted(os.listdir(self.emails_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.emails_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse email structure
                email_data = self._parse_email(content, filename)
                emails.append(email_data)
        
        return emails
    
    def _parse_email(self, content: str, filename: str) -> Dict[str, Any]:
        """Parse email content into structured format"""
        lines = content.strip().split('\n')
        
        email_data = {
            'filename': filename,
            'subject': '',
            'from_email': '',
            'to_email': '',
            'body': '',
            'full_content': content
        }
        
        # Extract subject
        for i, line in enumerate(lines):
            if line.startswith('Subject:'):
                email_data['subject'] = line.replace('Subject:', '').strip()
                break
        
        # Extract sender and receiver
        for line in lines:
            if line.startswith('From:'):
                email_data['from_email'] = line.replace('From:', '').strip()
            elif line.startswith('To:'):
                email_data['to_email'] = line.replace('To:', '').strip()
        
        # Extract body (everything after the greeting line)
        body_start = -1
        for i, line in enumerate(lines):
            if any(greeting in line.lower() for greeting in ['dear', 'hello', 'hi', 'hey']):
                body_start = i + 1
                break
        
        if body_start == -1:
            # Fallback: look for first empty line after headers
            for i, line in enumerate(lines):
                if not line.strip() and i > 3:
                    body_start = i + 1
                    break
        
        if body_start != -1:
            body_lines = []
            for line in lines[body_start:]:
                # Stop at signature lines
                if any(sig in line.lower() for sig in ['thanks', 'sincerely', 'regards', 'best']):
                    break
                if line.strip():
                    body_lines.append(line.strip())
            
            email_data['body'] = ' '.join(body_lines)
        
        return email_data
    
    def chunk_emails(self, emails: List[Dict[str, Any]]) -> List[EmailChunk]:
        """Chunk emails into smaller pieces for processing"""
        chunks = []
        
        for email in emails:
            # Create different types of chunks for better retrieval
            
            # 1. Subject chunk (important for topic-based queries)
            if email['subject']:
                subject_chunk = EmailChunk(
                    content=f"Subject: {email['subject']}",
                    metadata={
                        'type': 'subject',
                        'filename': email['filename'],
                        'from_email': email['from_email'],
                        'to_email': email['to_email']
                    },
                    chunk_id=f"{email['filename']}_subject"
                )
                chunks.append(subject_chunk)
            
            # 2. Body chunks (main content)
            body = email['body']
            if body:
                body_chunks = self._create_text_chunks(
                    body, email['filename'], 'body', email
                )
                chunks.extend(body_chunks)
            
            # 3. Full email chunk (for context-heavy queries)
            full_content = f"Subject: {email['subject']}\nFrom: {email['from_email']}\nTo: {email['to_email']}\n\n{email['body']}"
            full_chunk = EmailChunk(
                content=full_content,
                metadata={
                    'type': 'full_email',
                    'filename': email['filename'],
                    'from_email': email['from_email'],
                    'to_email': email['to_email']
                },
                chunk_id=f"{email['filename']}_full"
            )
            chunks.append(full_chunk)
        
        return chunks
    
    def _create_text_chunks(self, text: str, filename: str, chunk_type: str, email_data: Dict[str, Any]) -> List[EmailChunk]:
        """Create overlapping chunks from text"""
        if len(text) <= self.chunk_size:
            return [EmailChunk(
                content=text,
                metadata={
                    'type': chunk_type,
                    'filename': filename,
                    'from_email': email_data['from_email'],
                    'to_email': email_data['to_email']
                },
                chunk_id=f"{filename}_{chunk_type}_0"
            )]
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = max(
                    text.rfind('.', start, end + 50),
                    text.rfind('!', start, end + 50),
                    text.rfind('?', start, end + 50)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = EmailChunk(
                    content=chunk_text,
                    metadata={
                        'type': chunk_type,
                        'filename': filename,
                        'from_email': email_data['from_email'],
                        'to_email': email_data['to_email']
                    },
                    chunk_id=f"{filename}_{chunk_type}_{chunk_idx}"
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
