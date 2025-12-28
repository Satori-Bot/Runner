"""
Tests for Agent Runner backend service.
"""

import pytest
from unittest.mock import AsyncMock, patch
import json

# Import the module under test
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/backend")

from agent_runner import AgentRunner, Job, JobStatus


class TestAgentRunner:
    """Tests for AgentRunner class."""
    
    def test_validate_repo_path_valid(self):
        """Test valid repository paths."""
        assert AgentRunner._validate_repo_path("owner/repo") is True
        assert AgentRunner._validate_repo_path("my-org/my-repo") is True
        assert AgentRunner._validate_repo_path("user123/project_name") is True
        assert AgentRunner._validate_repo_path("a/b") is True
    
    def test_validate_repo_path_invalid(self):
        """Test invalid repository paths."""
        assert AgentRunner._validate_repo_path("") is False
        assert AgentRunner._validate_repo_path("repo") is False
        assert AgentRunner._validate_repo_path("owner/") is False
        assert AgentRunner._validate_repo_path("/repo") is False
        assert AgentRunner._validate_repo_path("owner/repo/extra") is False
        assert AgentRunner._validate_repo_path("owner repo") is False
    
    def test_validate_callback_url_valid(self):
        """Test valid callback URLs."""
        assert AgentRunner._validate_callback_url("https://example.com/webhook") is True
        assert AgentRunner._validate_callback_url("http://localhost:8000/callback") is True
        assert AgentRunner._validate_callback_url("https://api.example.com/v1/hook") is True
    
    def test_validate_callback_url_invalid(self):
        """Test invalid callback URLs."""
        assert AgentRunner._validate_callback_url("") is False
        assert AgentRunner._validate_callback_url("not-a-url") is False
        assert AgentRunner._validate_callback_url("ftp://example.com") is False
        assert AgentRunner._validate_callback_url("https:// example.com") is False  # space
    
    def test_job_to_dict(self):
        """Test Job serialization."""
        job = Job(
            job_id="job-test123",
            upstream_repo="owner/repo",
            prompt="Fix the bug",
            callback_url="https://example.com/webhook",
            status=JobStatus.PENDING,
        )
        
        result = job.to_dict()
        
        assert result["job_id"] == "job-test123"
        assert result["upstream_repo"] == "owner/repo"
        assert result["prompt"] == "Fix the bug"
        assert result["callback_url"] == "https://example.com/webhook"
        assert result["status"] == "pending"
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_job_status_values(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.FORKING.value == "forking"
        assert JobStatus.FORK_READY.value == "fork_ready"
        assert JobStatus.TRIGGERED.value == "triggered"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestAgentRunnerInstance:
    """Tests that require an AgentRunner instance."""
    
    @pytest.fixture
    def runner(self):
        """Create a test runner instance."""
        return AgentRunner(
            bot_token="test-token",
            runner_repo="test-org/Agent-Runner",
            bot_username="test-bot",
            webhook_secret="test-secret",
        )
    
    def test_init(self, runner):
        """Test AgentRunner initialization."""
        assert runner.bot_token == "test-token"
        assert runner.runner_repo == "test-org/Agent-Runner"
        assert runner.bot_username == "test-bot"
        assert runner.webhook_secret == "test-secret"
        assert runner.allow_insecure_webhooks is False
    
    def test_verify_webhook_signature_valid(self, runner):
        """Test valid webhook signature verification."""
        import hmac
        import hashlib
        
        payload = b'{"job_id": "test", "status": "completed"}'
        expected = "sha256=" + hmac.new(
            b"test-secret",
            payload,
            hashlib.sha256,
        ).hexdigest()
        
        assert runner.verify_webhook_signature(payload, expected) is True
    
    def test_verify_webhook_signature_invalid(self, runner):
        """Test invalid webhook signature verification."""
        payload = b'{"job_id": "test", "status": "completed"}'
        assert runner.verify_webhook_signature(payload, "sha256=invalid") is False
    
    def test_verify_webhook_signature_no_secret_insecure(self):
        """Test webhook verification with no secret in insecure mode."""
        runner = AgentRunner(
            bot_token="test-token",
            runner_repo="test-org/Agent-Runner",
            bot_username="test-bot",
            webhook_secret=None,
            allow_insecure_webhooks=True,
        )
        
        payload = b'{"job_id": "test"}'
        assert runner.verify_webhook_signature(payload, "") is True
    
    def test_verify_webhook_signature_no_secret_secure(self):
        """Test webhook verification with no secret in secure mode."""
        runner = AgentRunner(
            bot_token="test-token",
            runner_repo="test-org/Agent-Runner",
            bot_username="test-bot",
            webhook_secret=None,
            allow_insecure_webhooks=False,
        )
        
        payload = b'{"job_id": "test"}'
        assert runner.verify_webhook_signature(payload, "") is False
    
    def test_get_job_not_found(self, runner):
        """Test getting a non-existent job."""
        assert runner.get_job("nonexistent") is None
    
    def test_update_job_from_callback_not_found(self, runner):
        """Test updating a non-existent job."""
        result = runner.update_job_from_callback(
            job_id="nonexistent",
            status="completed",
        )
        assert result is None
    
    def test_update_job_from_callback_completed(self, runner):
        """Test updating a job to completed status."""
        # Create a job first
        job = Job(
            job_id="test-job",
            upstream_repo="owner/repo",
            prompt="Test",
        )
        runner._jobs["test-job"] = job
        
        # Update via callback
        result = runner.update_job_from_callback(
            job_id="test-job",
            status="completed",
            pr_url="https://github.com/owner/repo/pull/1",
        )
        
        assert result is not None
        assert result.status == JobStatus.COMPLETED
        assert result.pr_url == "https://github.com/owner/repo/pull/1"
    
    def test_update_job_from_callback_failed(self, runner):
        """Test updating a job to failed status."""
        job = Job(
            job_id="test-job",
            upstream_repo="owner/repo",
            prompt="Test",
        )
        runner._jobs["test-job"] = job
        
        result = runner.update_job_from_callback(
            job_id="test-job",
            status="failed",
            error="Something went wrong",
        )
        
        assert result is not None
        assert result.status == JobStatus.FAILED
        assert result.error == "Something went wrong"


class TestSubmitJob:
    """Tests for job submission (requires mocking)."""
    
    @pytest.fixture
    def runner(self):
        return AgentRunner(
            bot_token="test-token",
            runner_repo="test-org/Agent-Runner",
            bot_username="test-bot",
        )
    
    @pytest.mark.asyncio
    async def test_submit_job_invalid_repo(self, runner):
        """Test submitting a job with invalid repo path."""
        with pytest.raises(ValueError, match="Invalid repository path"):
            await runner.submit_job(
                upstream_repo="invalid",
                prompt="Test prompt",
            )
    
    @pytest.mark.asyncio
    async def test_submit_job_empty_prompt(self, runner):
        """Test submitting a job with empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await runner.submit_job(
                upstream_repo="owner/repo",
                prompt="",
            )
    
    @pytest.mark.asyncio
    async def test_submit_job_invalid_callback_url(self, runner):
        """Test submitting a job with invalid callback URL."""
        with pytest.raises(ValueError, match="Invalid callback_url"):
            await runner.submit_job(
                upstream_repo="owner/repo",
                prompt="Test prompt",
                callback_url="not-a-url",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
