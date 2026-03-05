import { Container, Text, Title } from '@mantine/core';

export default function ResultsPage() {
  return (
    <Container size="lg" py="xl">
      <Title order={1} mb="xl">
        Results History
      </Title>
      <Text c="dimmed">
        Job history and result browsing will be available in a future phase.
      </Text>
    </Container>
  );
}
