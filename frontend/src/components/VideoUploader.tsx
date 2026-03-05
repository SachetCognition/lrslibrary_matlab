import { Group, Image, Text } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { useState } from 'react';
import apiClient from '../api/client';
import type { VideoMetadata } from '../types';

interface VideoUploaderProps {
  onUpload: (videoId: string, metadata: VideoMetadata) => void;
}

export default function VideoUploader({ onUpload }: VideoUploaderProps) {
  const [uploading, setUploading] = useState(false);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDrop = async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiClient.post<VideoMetadata>(
        '/video/upload',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );

      setMetadata(response.data);
      onUpload(response.data.video_id, response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <Dropzone
        onDrop={handleDrop}
        loading={uploading}
        accept={['video/avi', 'video/mp4', 'video/x-msvideo']}
        maxSize={100 * 1024 * 1024}
      >
        <Group
          justify="center"
          gap="xl"
          mih={120}
          style={{ pointerEvents: 'none' }}
        >
          <div>
            <Text size="xl" inline>
              Drag video here or click to select
            </Text>
            <Text size="sm" c="dimmed" inline mt={7}>
              Supports .avi and .mp4 files (max 100MB)
            </Text>
          </div>
        </Group>
      </Dropzone>

      {error && (
        <Text c="red" mt="sm">
          {error}
        </Text>
      )}

      {metadata && (
        <Group mt="md" gap="lg">
          <Image
            src={`/api/video/${metadata.video_id}/frame/0`}
            alt="First frame preview"
            w={200}
            fit="contain"
          />
          <div>
            <Text size="sm">
              Resolution: {metadata.width} x {metadata.height}
            </Text>
            <Text size="sm">Frames: {metadata.nframes}</Text>
            <Text size="sm">FPS: {metadata.fps.toFixed(1)}</Text>
          </div>
        </Group>
      )}
    </div>
  );
}
