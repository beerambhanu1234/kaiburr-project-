import React from 'react';
import { Button, Popconfirm, Space, Table, Tag, Tooltip } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { PlayCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import type { Task } from '../types/task';

interface Props {
  data: Task[];
  loading?: boolean;
  onRun: (task: Task) => void;
  onDelete: (task: Task) => void;
}

export default function TaskTable({ data, loading, onRun, onDelete }: Props) {
  const columns: ColumnsType<Task> = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Command', dataIndex: 'command', key: 'command', ellipsis: true },
    {
      title: 'Last Run', dataIndex: 'lastRunAt', key: 'lastRunAt', width: 180,
      render: (v?: string) => v ? new Date(v).toLocaleString() : '-'
    },
    {
      title: 'Status', dataIndex: 'lastStatus', key: 'lastStatus', width: 120,
      render: (s?: Task['lastStatus']) => s ? <Tag color={s === 'SUCCESS' ? 'green' : s === 'FAILED' ? 'red' : 'blue'}>{s}</Tag> : '-'
    },
    {
      key: 'actions', title: 'Actions', width: 180,
      render: (_, record) => (
        <Space>
          <Tooltip title="Run">
            <Button icon={<PlayCircleOutlined />} onClick={() => onRun(record)}>
              Run
            </Button>
          </Tooltip>
          <Popconfirm title="Delete task?" onConfirm={() => onDelete(record)}>
            <Button danger icon={<DeleteOutlined />}>Delete</Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  return (
    <Table
      rowKey={r => r.id || r.name}
      dataSource={data}
      columns={columns}
      loading={loading}
      pagination={{ pageSize: 8 }}
    />
  );
}
