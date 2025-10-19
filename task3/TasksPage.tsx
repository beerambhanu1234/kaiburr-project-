import React, { useEffect, useMemo, useState } from 'react';
import { Layout, Typography, Row, Col, Card, Input, message, Drawer, Space } from 'antd';
import TaskForm from '../components/TaskForm';
import TaskTable from '../components/TaskTable';
import { createTask, deleteTask, listTasks, runTask, searchTasksByName } from '../api/client';
import type { Task } from '../types/task';

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;

export default function TasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState('');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerOutput, setDrawerOutput] = useState<string>('');
  const [drawerTitle, setDrawerTitle] = useState<string>('Execution Output');

  const fetchTasks = async () => {
    setLoading(true);
    try {
      const data = search
        ? await searchTasksByName(search)
        : await listTasks();
      setTasks(data);
    } catch (e: any) {
      message.error(e?.message || 'Failed to fetch tasks');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTasks();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const t = setTimeout(fetchTasks, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  const onCreate = async (task: Omit<Task, 'id'>) => {
    setCreating(true);
    try {
      await createTask(task);
      message.success('Task created');
      await fetchTasks();
    } catch (e: any) {
      message.error(e?.message || 'Create failed');
    } finally {
      setCreating(false);
    }
  };

  const onDelete = async (task: Task) => {
    try {
      await deleteTask(task.id!);
      message.success('Task deleted');
      await fetchTasks();
    } catch (e: any) {
      message.error(e?.message || 'Delete failed');
    }
  };

  const onRun = async (task: Task) => {
    try {
      const res = await runTask(task.id!);
      const output = res?.output || JSON.stringify(res, null, 2);
      setDrawerTitle(`Execution Output - ${task.name}`);
      setDrawerOutput(output || 'No output');
      setDrawerOpen(true);
      await fetchTasks();
    } catch (e: any) {
      message.error(e?.message || 'Run failed');
    }
  };

  const headerExtra = useMemo(() => (
    <Input.Search
      placeholder="Search by name"
      allowClear
      onChange={(e) => setSearch(e.target.value)}
      style={{ maxWidth: 320 }}
    />
  ), []);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', borderBottom: '1px solid #f0f0f0' }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Title level={3} style={{ margin: 0 }}>Task Manager</Title>
            <Text type="secondary">Create, search, run, and delete tasks</Text>
          </Col>
          <Col>
            {headerExtra}
          </Col>
        </Row>
      </Header>
      <Content style={{ padding: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={8}>
            <Card title="Create Task">
              <TaskForm loading={creating} onSubmit={onCreate} />
            </Card>
          </Col>
          <Col xs={24} lg={16}>
            <Card title="Tasks" extra={<Text type="secondary">{tasks.length} items</Text>}>
              <TaskTable data={tasks} loading={loading} onRun={onRun} onDelete={onDelete} />
            </Card>
          </Col>
        </Row>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        <Space>
          <Text>API</Text>
          <Text code>http://localhost:8085</Text>
        </Space>
      </Footer>

      <Drawer
        title={drawerTitle}
        placement="right"
        width={520}
        onClose={() => setDrawerOpen(false)}
        open={drawerOpen}
      >
        <pre style={{ whiteSpace: 'pre-wrap' }}>{drawerOutput}</pre>
      </Drawer>
    </Layout>
  );
}
