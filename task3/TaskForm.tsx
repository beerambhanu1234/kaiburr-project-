import React from 'react';
import { Button, Form, Input, Space } from 'antd';
import type { Task } from '../types/task';

interface Props {
  loading?: boolean;
  onSubmit: (task: Omit<Task, 'id'>) => void;
}

export default function TaskForm({ loading, onSubmit }: Props) {
  const [form] = Form.useForm<Omit<Task, 'id'>>();

  const handleFinish = (values: Omit<Task, 'id'>) => {
    onSubmit(values);
    form.resetFields();
  };

  return (
    <Form form={form} layout="vertical" onFinish={handleFinish} autoComplete="off">
      <Form.Item label="Name" name="name" rules={[{ required: true, message: 'Please input task name' }]}>
        <Input placeholder="e.g., Backup DB" allowClear />
      </Form.Item>
      <Form.Item label="Command" name="command" rules={[{ required: true, message: 'Please input command' }]}>
        <Input placeholder="e.g., echo Hello" allowClear />
      </Form.Item>
      <Form.Item label="Schedule (optional)" name="schedule">
        <Input placeholder="cron or description" allowClear />
      </Form.Item>
      <Space>
        <Button type="primary" htmlType="submit" loading={loading}>Create Task</Button>
        <Button htmlType="button" onClick={() => form.resetFields()} disabled={loading}>Reset</Button>
      </Space>
    </Form>
  );
}
