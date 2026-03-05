import { Container, Stepper, Title } from '@mantine/core';
import { useState } from 'react';
import AlgorithmSelector from '../components/AlgorithmSelector';
import MethodSelector from '../components/MethodSelector';
import PerformanceStats from '../components/PerformanceStats';
import ProcessButton from '../components/ProcessButton';
import ResultViewer from '../components/ResultViewer';
import VideoUploader from '../components/VideoUploader';
import type { JobStatus, VideoMetadata } from '../types';

export default function ProcessPage() {
  const [methodId, setMethodId] = useState<string | null>(null);
  const [algorithmId, setAlgorithmId] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<VideoMetadata | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobStatus | null>(null);

  const activeStep = jobId
    ? 4
    : videoId
      ? 3
      : algorithmId
        ? 2
        : methodId
          ? 1
          : 0;

  const handleMethodSelect = (id: string) => {
    setMethodId(id);
    setAlgorithmId(null);
    setJobId(null);
    setJobResult(null);
  };

  const handleAlgorithmSelect = (id: string) => {
    setAlgorithmId(id);
    setJobId(null);
    setJobResult(null);
  };

  const handleVideoUpload = (id: string, meta: VideoMetadata) => {
    setVideoId(id);
    setVideoMeta(meta);
    setJobId(null);
    setJobResult(null);
  };

  const handleJobComplete = (id: string, status: JobStatus) => {
    setJobId(id);
    setJobResult(status);
  };

  return (
    <Container size="lg" py="xl">
      <Title order={1} mb="xl">
        Video Decomposition
      </Title>

      <Stepper active={activeStep} mb="xl">
        <Stepper.Step label="Method" description="Select category" />
        <Stepper.Step label="Algorithm" description="Choose algorithm" />
        <Stepper.Step label="Video" description="Upload video" />
        <Stepper.Step label="Process" description="Run decomposition" />
        <Stepper.Step label="Results" description="View output" />
      </Stepper>

      <MethodSelector onSelect={handleMethodSelect} />

      {methodId && (
        <div style={{ marginTop: '2rem' }}>
          <AlgorithmSelector
            methodId={methodId}
            onSelect={handleAlgorithmSelect}
          />
        </div>
      )}

      {algorithmId && (
        <div style={{ marginTop: '2rem' }}>
          <VideoUploader onUpload={handleVideoUpload} />
        </div>
      )}

      {videoId && (
        <div style={{ marginTop: '2rem' }}>
          <ProcessButton
            methodId={methodId}
            algorithmId={algorithmId}
            videoId={videoId}
            onComplete={handleJobComplete}
          />
        </div>
      )}

      {jobId && (
        <div style={{ marginTop: '2rem' }}>
          <ResultViewer jobId={jobId} />
        </div>
      )}

      {jobResult?.status === 'complete' && (
        <div style={{ marginTop: '2rem' }}>
          <PerformanceStats
            cputime={jobResult.cputime ?? null}
            algorithmName={`${methodId} / ${algorithmId}`}
            matrixDimensions={
              videoMeta
                ? `${videoMeta.height * videoMeta.width} x ${videoMeta.nframes}`
                : 'N/A'
            }
          />
        </div>
      )}
    </Container>
  );
}
