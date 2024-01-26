from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="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"
)


# apply the pipeline to an audio file
diarization = pipeline("20240126_094429.m4a")

# dump the diarization output to disk using RTTM format
with open("./data/bronze/20240126_094429.m4a.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
