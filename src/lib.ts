import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream, promises as fs } from 'node:fs';
import { basename, extname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Redis } from 'ioredis';

export const dataDir = resolve(process.env.DATA_DIR || './data');
export const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379', { maxRetriesPerRequest: null });
export const maxFiles = Number(process.env.MAX_BATCH_FILES || 2000);
export const maxBytes = Number(process.env.MAX_FILE_BYTES || 20 * 1024 * 1024);
export type State = 'uploaded'|'queued'|'processing'|'completed'|'needs_review'|'failed'|'cancelled';
export type Settings = { background: string; preset: 'none'|'natural'|'enhanced'; width: number; height: number; crop: boolean; format: 'jpg'|'png'; quality: number; dpi?:number; autoDetect?:boolean; enhance?:boolean; sharpness?:number; countryProfile?:string };
export type FileRecord = { id:string; batchId:string; sourceName:string; outputName:string; inputPath:string; outputPath:string; state:State; error?:string; attempts:string; customSettings?:string; createdAt:string; updatedAt:string };
export const batchPath = (id:string) => join(dataDir, 'batches', id);
export async function init(){ await fs.mkdir(join(dataDir, 'batches'), {recursive:true}); }
export const safeName=(name:string)=>basename(name).replace(/[\\/\0]/g,'_');
export function outputName(source:string, format:string, index:number) { const ext=extname(source); const stem=source.slice(0, source.length-ext.length) || source; return `${stem}${index ? ` (${index + 1})` : ''}.${format}`; }
export async function saveRecord(r:FileRecord){ r.updatedAt=new Date().toISOString(); await redis.hmset(`file:${r.id}`, r as unknown as Record<string,string>); await redis.sadd(`batch:${r.batchId}:files`,r.id); }
export async function file(id:string){ const f=await redis.hgetall(`file:${id}`); return Object.keys(f).length ? f as FileRecord : null; }
export async function files(batchId:string){ const ids=await redis.smembers(`batch:${batchId}:files`); return (await Promise.all(ids.map(file))).filter(Boolean) as FileRecord[]; }
export async function batch(id:string){ const b=await redis.hgetall(`batch:${id}`); return Object.keys(b).length ? b : null; }
export async function saveBatch(id:string, values:Record<string,string>){ await redis.hmset(`batch:${id}`, values); }
export function digest(s:string){ return createHash('sha256').update(s).digest('hex'); }
export async function writeUpload(stream:NodeJS.ReadableStream,path:string){ await pipeline(stream,createWriteStream(path,{flags:'wx'})); }
export { randomUUID, fs, join, createReadStream, createWriteStream };
