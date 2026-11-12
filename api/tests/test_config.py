import pytest
from pydantic import ValidationError

from api.app.config import Settings


def test_origins_are_split_and_trimmed():
    s = Settings(cors_allow_origins="https://a.example.com, http://localhost:5173")
    assert s.cors_origins == ["https://a.example.com", "http://localhost:5173"]


@pytest.mark.parametrize(
    "bad",
    [
        "*",                        # fail-open wildcard
        "",                         # empty
        " , ",                      # only separators
        "a.example.com",            # missing scheme
        "https://a.example.com/",   # trailing slash never matches a browser Origin
    ],
)
def test_bad_origins_are_rejected(bad):
    with pytest.raises(ValidationError):
        Settings(cors_allow_origins=bad)


def test_missing_origins_refuses_to_start(monkeypatch):
    # Remove the conftest default and ignore .env, simulating a misconfigured deploy.
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)