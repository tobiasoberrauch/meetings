from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="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"
)
