import {
  Badge,
  Container,
  Group,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { useState } from 'react';
import { useAlgorithms } from '../hooks/useAlgorithms';

const SPEED_COLORS: Record<number, string> = {
  1: 'green',
  2: 'lime',
  3: 'yellow',
  4: 'orange',
  5: 'red',
};

export default function HomePage() {
  const { data: algorithms, isLoading } = useAlgorithms();
  const [search, setSearch] = useState('');

  const filtered = (algorithms ?? []).filter(
    (algo) =>
      algo.algorithm_id.toLowerCase().includes(search.toLowerCase()) ||
      algo.name.toLowerCase().includes(search.toLowerCase()) ||
      algo.method_id.toLowerCase().includes(search.toLowerCase())
  );

  // Group by method_id
  const grouped: Record<string, typeof filtered> = {};
  for (const algo of filtered) {
    if (!grouped[algo.method_id]) {
      grouped[algo.method_id] = [];
    }
    grouped[algo.method_id].push(algo);
  }

  return (
    <Container size="lg" py="xl">
      <Title order={1} mb="sm">
        LRSLibrary
      </Title>
      <Text c="dimmed" mb="xl">
        Low-Rank and Sparse Decomposition Library — migrated from MATLAB.
        Currently {algorithms?.length ?? 0} algorithms across{' '}
        {Object.keys(grouped).length} categories.
      </Text>

      <TextInput
        placeholder="Search algorithms..."
        value={search}
        onChange={(e) => setSearch(e.currentTarget.value)}
        mb="lg"
      />

      {isLoading ? (
        <Text>Loading algorithms...</Text>
      ) : (
        Object.entries(grouped)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([methodId, algos]) => (
            <div key={methodId} style={{ marginBottom: '2rem' }}>
              <Group mb="xs">
                <Title order={3}>{methodId}</Title>
                <Badge color="blue" variant="light">
                  {algos.length}
                </Badge>
              </Group>
              <Table striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Algorithm ID</Table.Th>
                    <Table.Th>Name</Table.Th>
                    <Table.Th>Speed</Table.Th>
                    <Table.Th>Type</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {algos.map((algo) => (
                    <Table.Tr key={algo.algorithm_id}>
                      <Table.Td>
                        <Text fw={500}>{algo.algorithm_id}</Text>
                      </Table.Td>
                      <Table.Td>{algo.name}</Table.Td>
                      <Table.Td>
                        <Badge
                          color={SPEED_COLORS[algo.speed_class] ?? 'gray'}
                          variant="light"
                        >
                          {algo.speed_class}
                        </Badge>
                      </Table.Td>
                      <Table.Td>
                        {algo.is_tensor ? (
                          <Badge color="violet" variant="light">
                            Tensor
                          </Badge>
                        ) : (
                          <Badge color="gray" variant="light">
                            Matrix
                          </Badge>
                        )}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </div>
          ))
      )}
    </Container>
  );
}
