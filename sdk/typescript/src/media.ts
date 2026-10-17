import type { MediaInput } from './index.js';

function media(type: MediaInput['type'], data: Uint8Array | string): MediaInput {
  if (data instanceof Uint8Array) {
    // Bounded chunks avoid the argument limit of spreading a whole image/audio file.
    const chunks: string[]=[];
    for(let offset=0;offset<data.length;offset+=8192)chunks.push(String.fromCharCode(...data.subarray(offset,offset+8192)));
    data=btoa(chunks.join(''));
  }
  if(typeof data!=='string'||!data.trim()||/^https?:\/\//i.test(data))throw new TypeError('media data must be bytes, base64 or a data URL');
  return {type,data};
}
/** Inline image; never fetches a URL or reads a file. */
export function image(data: Uint8Array | string): MediaInput {return media('image',data);}
/** Inline audio; never fetches a URL or reads a file. */
export function audio(data: Uint8Array | string): MediaInput {return media('audio',data);}
