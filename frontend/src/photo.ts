// Doit rester aligné sur QRALIA_MAX_UPLOAD_MB côté API (25 par défaut).
export const MAX_UPLOAD_MB = 25;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

const COMPRESS_ABOVE_BYTES = 4 * 1024 * 1024;
// 3200 px sur le grand côté suffisent largement à l'OCR d'une facture A4.
const MAX_SIDE = 3200;

/**
 * Réduit une grosse photo de téléphone avant l'envoi : plus rapide à envoyer et
 * à lire, sans perte utile. Les PDF et les petites images sont envoyés tels quels.
 */
export async function prepareUpload(file: File): Promise<File> {
  if (!/^image\/(jpeg|png)$/.test(file.type) || file.size <= COMPRESS_ABOVE_BYTES) return file;
  try {
    // « from-image » applique l'orientation EXIF : une photo prise en portrait le reste.
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob || blob.size >= file.size) return file;
    return new File([blob], file.name.replace(/\.(png|jpe?g)$/i, ".jpg"), { type: "image/jpeg" });
  } catch {
    // Navigateur trop ancien ou image illisible : l'API décidera.
    return file;
  }
}
