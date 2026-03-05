import { Grid, Image, Slider, Text, Title } from '@mantine/core';
import { useState } from 'react';
import { useJobStatus } from '../hooks/useJob';

interface ResultViewerProps {
  jobId: string;
}

const PANELS = [
  { key: 'input', label: 'Input' },
  { key: 'L', label: 'Low-Rank (L)' },
  { key: 'S', label: 'Sparse (S)' },
  { key: 'O', label: 'Outliers (O)' },
] as const;

export default function ResultViewer({ jobId }: ResultViewerProps) {
  const { data: jobStatus, isLoading } = useJobStatus(jobId);
  const [frameNum, setFrameNum] = useState(0);

  if (isLoading || !jobStatus) {
    return <Text>Loading results...</Text>;
  }

  if (jobStatus.status === 'pending' || jobStatus.status === 'running') {
    return <Text>Processing... ({Math.round(jobStatus.progress * 100)}%)</Text>;
  }

  if (jobStatus.status === 'failed') {
    return <Text c="red">Job failed: {jobStatus.error ?? 'Unknown error'}</Text>;
  }

  const resultUrls = jobStatus.result_urls ?? {};

  return (
    <div>
      <Title order={3} mb="md">
        Decomposition Results
      </Title>

      <Grid>
        {PANELS.map((panel) => (
          <Grid.Col span={6} key={panel.key}>
            <Text fw={500} mb="xs">
              {panel.label}
            </Text>
            {panel.key === 'input' ? (
              <Text size="sm" c="dimmed">
                Original input video
              </Text>
            ) : (
              <Image
                src={`${resultUrls[panel.key]}/frame/${frameNum}`}
                alt={panel.label}
                fit="contain"
                fallbackSrc="data:image/svg+xml;charset=utf-8,<svg xmlns='http://www.w3.org/2000/svg'/>"
              />
            )}
          </Grid.Col>
        ))}
      </Grid>

      <Text mt="md" mb="xs">
        Frame: {frameNum}
      </Text>
      <Slider
        value={frameNum}
        onChange={setFrameNum}
        min={0}
        max={99}
        step={1}
        label={(value) => `Frame ${value}`}
      />
    </div>
  );
}
