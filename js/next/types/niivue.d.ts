// NiiVue is loaded from unpkg through next.html's import map, so it has no node_modules types.
// js/next/viewer/ uses this much of its API (NiiVue 0.69).
export class NVImage {
  constructor(dataBuffer?: ArrayBufferLike | null, name?: string, colormap?: string, opacity?: number,
    pairedImgData?: ArrayBuffer | null, cal_min?: number, cal_max?: number, trustCalMinMax?: boolean);
  static new(dataBuffer: ArrayBufferLike | null, name?: string, colormap?: string, opacity?: number,
    pairedImgData?: ArrayBuffer | null, cal_min?: number, cal_max?: number, trustCalMinMax?: boolean,
    percentileFrac?: number, ignoreZeroVoxels?: boolean): Promise<NVImage>;
  static createNiftiArray(dims?: number[], pixDims?: number[], affine?: number[], datatypeCode?: number,
    img?: ArrayBufferView): Uint8Array;
  name: string;
  cal_min: number;
  cal_max: number;
  colormap: string;
}
export class Niivue {
  constructor(options?: Record<string, unknown>);
  attachToCanvas(canvas: HTMLCanvasElement, isAntiAlias?: boolean | null): Promise<this>;
  addVolume(volume: NVImage): void;
  removeVolume(volume: NVImage): void;
  setSliceType(type: number): this;
  updateGLVolume(): void;
  drawScene(): void;
  volumes: NVImage[];
  opts: Record<string, any>;
  loadDrawing(drawing: NVImage): boolean;
  setDrawingEnabled(on: boolean): void;
  setDrawOpacity(opacity: number): void;
  setPenValue(value: number, isFilledPen?: boolean): void;
  closeDrawing(): void;
  saveImage(options: { filename: string, isSaveDrawing?: boolean, volumeByIndex?: number }): Promise<Uint8Array | boolean>;
  sliceTypeAxial: number;
  sliceTypeCoronal: number;
  sliceTypeSagittal: number;
  sliceTypeMultiplanar: number;
  sliceTypeRender: number;
}
export const SLICE_TYPE: { AXIAL: number; CORONAL: number; SAGITTAL: number; MULTIPLANAR: number; RENDER: number };
