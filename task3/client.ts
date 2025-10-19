import axios from 'axios';
import { API_BASE } from '../config';
import type { Task } from '../types/task';

const http = axios.create({ baseURL: API_BASE, timeout: 20000 });

export async function listTasks(): Promise<Task[]> {
  const { data } = await http.get('/tasks');
  return data;
}

export async function getTaskById(id: string): Promise<Task> {
  const { data } = await http.get('/tasks', { params: { id } });
  return data;
}

export async function searchTasksByName(name: string): Promise<Task[]> {
  const { data } = await http.get('/tasks/search', { params: { name } });
  return data;
}

export async function createTask(task: Omit<Task, 'id'>): Promise<Task> {
  const { data } = await http.put('/tasks', task);
  return data;
}

export async function deleteTask(id: string): Promise<void> {
  await http.delete(`/tasks/${id}`);
}

export async function runTask(id: string): Promise<{ output?: string } & Partial<Task>> {
  const { data } = await http.put(`/tasks/${id}/executions`);
  return data;
}
