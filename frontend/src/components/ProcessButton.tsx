import { Button, Progress, Text } from '@mantine/core';
import { useSubmitJob } from '../hooks/useJob';
import type { JobStatus } from '../types';

interface ProcessButtonProps {
  methodId: string | null;
  algorithmId: string | null;
  videoId: string | null;
  onComplete: (jobId: string, status: JobStatus) => void;
}

export default function ProcessButton({
  methodId,
  algorithmId,
  videoId,
  onComplete,
}: ProcessButtonProps) {
  const submitJob = useSubmitJob();

  const isDisabled = !methodId || !algorithmId || !videoId;

  const handleClick = () => {
    if (!methodId || !algorithmId || !videoId) return;

    submitJob.mutate(
      {
        method_id: methodId,
        algorithm_id: algorithmId,
        video_id: videoId,
      },
      {
        onSuccess: (data) => {
          onComplete(data.job_id, data);
        },
      }
    );
  };

  return (
    <div>
      <Button
        size="lg"
        onClick={handleClick}
        loading={submitJob.isPending}
        disabled={isDisabled}
      >
        Run Decomposition
      </Button>

      {submitJob.isPending && (
        <Progress value={50} animated mt="md" />
      )}

      {submitJob.isError && (
        <Text c="red" mt="sm">
          Error: {submitJob.error?.message ?? 'Job submission failed'}
        </Text>
      )}
    </div>
  );
}
