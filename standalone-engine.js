/**
 * DocScan Study PDF - Core Computer Vision, IndexedDB & jsPDF Engine (100% Real Pixel Manipulation)
 * Bao gồm:
 * 1. iOS Safari RAM Protection (Max 2200px, JPEG 0.92)
 * 2. Auto Document Edge Detection (Luminance + Gaussian 3x3 + Sobel Gradient + Adaptive Page Mask)
 * 3. 4-Point Perspective Homography Warp (Gauss-Jordan 8x8 + Bilinear Interpolation)
 * 4. Rotation (90°/180°/270° + Fine Straighten -15°..+15°)
 * 5. Study Document Enhancer (Magic Color Local Shadow Removal, B&W High Contrast, Grayscale, Original)
 * 6. IndexedDB Storage Engine & Multi-page jsPDF Generator + iOS Share Sheet
 */
(function (global) {
  const IOS_MAX_DIMENSION = 2200;
  const IOS_JPEG_QUALITY = 0.92;

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function clampByte(v) {
    return v < 0 ? 0 : v > 255 ? 255 : v | 0;
  }

  function loadImageElement(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (err) => reject(err);
      img.src = src;
    });
  }

  function getDefaultFullQuad() {
    return {
      tl: { x: 0.0, y: 0.0 },
      tr: { x: 1.0, y: 0.0 },
      br: { x: 1.0, y: 1.0 },
      bl: { x: 0.0, y: 1.0 }
    };
  }

  function getDefaultInsetQuad() {
    return {
      tl: { x: 0.05, y: 0.05 },
      tr: { x: 0.95, y: 0.05 },
      br: { x: 0.95, y: 0.95 },
      bl: { x: 0.05, y: 0.95 }
    };
  }

  function computeQuadNormalizedArea(q) {
    const x1 = q.tl.x, y1 = q.tl.y;
    const x2 = q.tr.x, y2 = q.tr.y;
    const x3 = q.br.x, y3 = q.br.y;
    const x4 = q.bl.x, y4 = q.bl.y;
    return 0.5 * Math.abs(
      x1 * y2 - y1 * x2 +
      x2 * y3 - y2 * x3 +
      x3 * y4 - y3 * x4 +
      x4 * y1 - y4 * x1
    );
  }

  /**
   * MODULE 1: iOS Safari RAM Protection
   */
  async function compressForIOSMemoryGuard(fileOrDataUrl, maxDim = IOS_MAX_DIMENSION, quality = IOS_JPEG_QUALITY) {
    let rawSrc = '';
    let revokeNeeded = false;

    if (typeof fileOrDataUrl === 'string') {
      rawSrc = fileOrDataUrl;
    } else {
      rawSrc = URL.createObjectURL(fileOrDataUrl);
      revokeNeeded = true;
    }

    try {
      const img = await loadImageElement(rawSrc);
      let targetW = img.naturalWidth || img.width;
      let targetH = img.naturalHeight || img.height;

      const longestSide = Math.max(targetW, targetH);
      if (longestSide > maxDim) {
        const scale = maxDim / longestSide;
        targetW = Math.round(targetW * scale);
        targetH = Math.round(targetH * scale);
      }

      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, targetW, targetH);

      const dataUrl = canvas.toDataURL('image/jpeg', quality);
      canvas.width = 0;
      canvas.height = 0;

      return { dataUrl, width: targetW, height: targetH };
    } finally {
      if (revokeNeeded && rawSrc) {
        URL.revokeObjectURL(rawSrc);
      }
    }
  }

  /**
   * MODULE 2.1: Thuật toán Tự Động Bám Sát Viền Trang Sách & Cắt Bỏ Nền Bàn / Trang Bên Cạnh (Inside-Out Book Page & Spine Detector)
   * 1. Xóa chữ giữ nền bằng Phép Đóng Hình Thái Học (Morphological Closing = Dilation + Erosion 9x9)
   *    giúp cả trang sách trở thành mặt phẳng đồng nhất, làm nổi bật mép giấy, bìa sách, nền bàn và rãnh gáy sách.
   * 2. Cô lập Khối Chữ Trang Chính & Phát hiện Rãnh Gáy Sách (Spine / Adjacent Page Separator)
   *    để tách biệt hoàn toàn trang sách chính ở giữa với phần dư của trang sách bên cạnh.
   * 3. Quét Từ Trong Ra Ngoài (Inside-Out Edge Locking): Xuất phát từ ngay sát ngoài khối chữ trang chính quét ra 4 phía
   *    để bắt dính vào ĐƯỜNG BIÊN VẬT LÝ ĐẦU TIÊN (mép trang giấy hoặc rãnh gáy sách), loại bỏ 100% nền bàn, môi trường xung quanh
   *    và phần dư của trang sách bên cạnh mà không bao giờ cắt phạm vào chữ!
   */
  function detectDocumentCorners(imgOrCanvas) {
    const analysisSize = 360;
    const srcW = imgOrCanvas.naturalWidth || imgOrCanvas.width;
    const srcH = imgOrCanvas.naturalHeight || imgOrCanvas.height;

    const scale = Math.min(1, analysisSize / Math.max(srcW, srcH));
    const w = Math.max(80, Math.round(srcW * scale));
    const h = Math.max(80, Math.round(srcH * scale));

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    const ctx = tempCanvas.getContext('2d');
    if (!ctx) return getDefaultFullQuad();

    ctx.drawImage(imgOrCanvas, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;

    const numPixels = w * h;
    const lum = new Float32Array(numPixels);
    const rArr = new Float32Array(numPixels);
    const gArr = new Float32Array(numPixels);
    const bArr = new Float32Array(numPixels);

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      rArr[p] = r;
      gArr[p] = g;
      bArr[p] = b;
      lum[p] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // GIAI ĐOẠN 1: Phép Đóng Hình Thái Học (Morphological Closing = Dilation + Erosion bán kính 4px)
    const radius = 4;
    const tmpHoriz = new Float32Array(numPixels);
    const dilatedLum = new Float32Array(numPixels);

    for (let y = 0; y < h; y++) {
      const rowOff = y * w;
      for (let x = 0; x < w; x++) {
        let maxV = 0;
        const x0 = Math.max(0, x - radius);
        const x1 = Math.min(w - 1, x + radius);
        for (let xx = x0; xx <= x1; xx++) {
          const v = lum[rowOff + xx];
          if (v > maxV) maxV = v;
        }
        tmpHoriz[rowOff + x] = maxV;
      }
    }

    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(h - 1, y + radius);
      for (let x = 0; x < w; x++) {
        let maxV = 0;
        for (let yy = y0; yy <= y1; yy++) {
          const v = tmpHoriz[yy * w + x];
          if (v > maxV) maxV = v;
        }
        dilatedLum[y * w + x] = maxV;
      }
    }

    const closedLum = new Float32Array(numPixels);
    for (let y = 0; y < h; y++) {
      const rowOff = y * w;
      for (let x = 0; x < w; x++) {
        let minV = 255;
        const x0 = Math.max(0, x - radius);
        const x1 = Math.min(w - 1, x + radius);
        for (let xx = x0; xx <= x1; xx++) {
          const v = dilatedLum[rowOff + xx];
          if (v < minV) minV = v;
        }
        tmpHoriz[rowOff + x] = minV;
      }
    }

    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(h - 1, y + radius);
      for (let x = 0; x < w; x++) {
        let minV = 255;
        for (let yy = y0; yy <= y1; yy++) {
          const v = tmpHoriz[yy * w + x];
          if (v < minV) minV = v;
        }
        closedLum[y * w + x] = minV;
      }
    }

    // Lấy mẫu màu sắc & độ sáng chuẩn của giấy ở tâm trang sách chính
    let refLumSum = 0, refRSum = 0, refGSum = 0, refBSum = 0, refCnt = 0;
    const cx0 = Math.floor(w * 0.30);
    const cx1 = Math.floor(w * 0.70);
    const cy0 = Math.floor(h * 0.25);
    const cy1 = Math.floor(h * 0.75);

    for (let y = cy0; y <= cy1; y += 2) {
      for (let x = cx0; x <= cx1; x += 2) {
        const idx = y * w + x;
        if (dilatedLum[idx] - lum[idx] < 18) {
          refLumSum += closedLum[idx];
          refRSum += rArr[idx];
          refGSum += gArr[idx];
          refBSum += bArr[idx];
          refCnt++;
        }
      }
    }

    const refLum = refCnt > 0 ? refLumSum / refCnt : 205;
    const refR = refCnt > 0 ? refRSum / refCnt : 205;
    const refG = refCnt > 0 ? refGSum / refCnt : 205;
    const refB = refCnt > 0 ? refBSum / refCnt : 205;

    const colorDist = new Float32Array(numPixels);
    const isPaperSurface = new Uint8Array(numPixels);
    const minPaperLum = Math.max(88, refLum * 0.62);
    const maxPaperColorDist = 52;

    for (let p = 0; p < numPixels; p++) {
      const lumRatio = closedLum[p] > 40 ? refLum / Math.max(60, dilatedLum[p]) : 1.0;
      const clampedRatio = clamp(lumRatio, 0.7, 1.45);
      const dr = rArr[p] * clampedRatio - refR;
      const dg = gArr[p] * clampedRatio - refG;
      const db = bArr[p] * clampedRatio - refB;
      const cd = Math.sqrt(dr * dr + dg * dg + db * db);
      colorDist[p] = cd;

      if (closedLum[p] >= minPaperLum && cd <= maxPaperColorDist) {
        isPaperSurface[p] = 1;
      }
    }

    // GIAI ĐOẠN 2: Nhận diện Nét Chữ (Ink Map) & Phát hiện Rãnh Gáy Sách / Trang Sách Bên Cạnh
    const inkDiffThresh = Math.max(20, refLum * 0.13);
    const rawInk = new Uint8Array(numPixels);
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const idx = y * w + x;
        if (isPaperSurface[idx] && dilatedLum[idx] - lum[idx] >= inkDiffThresh) {
          rawInk[idx] = 1;
        }
      }
    }

    const cleanInk = new Uint8Array(numPixels);
    const colInk = new Uint16Array(w);
    const rowInk = new Uint16Array(h);

    for (let y = Math.floor(h * 0.05); y < Math.floor(h * 0.95); y++) {
      for (let x = Math.floor(w * 0.04); x < Math.floor(w * 0.96); x++) {
        const idx = y * w + x;
        if (!rawInk[idx]) continue;
        const nb =
          rawInk[idx - w - 1] + rawInk[idx - w] + rawInk[idx - w + 1] +
          rawInk[idx - 1] + rawInk[idx + 1] +
          rawInk[idx + w - 1] + rawInk[idx + w] + rawInk[idx + w + 1];
        if (nb >= 2) {
          cleanInk[idx] = 1;
          colInk[x]++;
          rowInk[y]++;
        }
      }
    }

    const colMeanLum = new Float32Array(w);
    const colEdgeStrength = new Float32Array(w);
    const yBand0 = Math.floor(h * 0.18);
    const yBand1 = Math.floor(h * 0.82);
    const bandHeight = Math.max(1, yBand1 - yBand0);

    for (let x = 3; x < w - 3; x++) {
      let sumL = 0, sumEdge = 0;
      for (let y = yBand0; y < yBand1; y++) {
        const idx = y * w + x;
        sumL += closedLum[idx];
        sumEdge += Math.abs(closedLum[idx + 2] - closedLum[idx - 2]) + Math.abs(colorDist[idx + 2] - colorDist[idx - 2]) * 0.7;
      }
      colMeanLum[x] = sumL / bandHeight;
      colEdgeStrength[x] = sumEdge / bandHeight;
    }

    let leftPageLimit = Math.floor(w * 0.015);
    let rightPageLimit = Math.floor(w * 0.985);

    const minMarginGapCols = Math.max(4, Math.round(w * 0.028));
    let mainTextLeft = Math.floor(w * 0.12);
    let mainTextRight = Math.floor(w * 0.88);

    for (let x = Math.floor(w * 0.45); x >= Math.floor(w * 0.04); x--) {
      let emptyRun = 0;
      for (let k = 0; k < minMarginGapCols && x - k >= 0; k++) {
        if (colInk[x - k] <= 1) emptyRun++;
      }
      if (emptyRun >= minMarginGapCols) {
        mainTextLeft = x + 1;
        break;
      }
    }

    for (let x = Math.floor(w * 0.55); x <= Math.floor(w * 0.96); x++) {
      let emptyRun = 0;
      for (let k = 0; k < minMarginGapCols && x + k < w; k++) {
        if (colInk[x + k] <= 1) emptyRun++;
      }
      if (emptyRun >= minMarginGapCols) {
        mainTextRight = x - 1;
        break;
      }
    }

    let bestLeftBoundaryX = Math.max(Math.floor(w * 0.015), mainTextLeft - Math.round(w * 0.055));
    let maxLeftSpineScore = -1;
    for (let x = mainTextLeft - 2; x >= Math.floor(w * 0.02); x--) {
      const lumDrop = Math.max(0, refLum - colMeanLum[x]);
      const localValley = Math.max(0, (colMeanLum[Math.min(w - 1, x + 5)] + colMeanLum[Math.max(0, x - 5)]) * 0.5 - colMeanLum[x]);
      const score = colEdgeStrength[x] * 1.4 + lumDrop * 0.7 + localValley * 2.5 - colInk[x] * 4;
      if (score > maxLeftSpineScore && score > 14) {
        maxLeftSpineScore = score;
        bestLeftBoundaryX = x;
      }
      if (x < mainTextLeft - minMarginGapCols && colInk[x] >= 4 && colInk[x - 1] >= 4) {
        leftPageLimit = x + 2;
        break;
      }
    }
    leftPageLimit = Math.max(leftPageLimit, Math.min(bestLeftBoundaryX, mainTextLeft - Math.round(w * 0.02)));

    let bestRightBoundaryX = Math.min(Math.floor(w * 0.985), mainTextRight + Math.round(w * 0.055));
    let maxRightSpineScore = -1;
    for (let x = mainTextRight + 2; x <= Math.floor(w * 0.98); x++) {
      const lumDrop = Math.max(0, refLum - colMeanLum[x]);
      const localValley = Math.max(0, (colMeanLum[Math.min(w - 1, x + 5)] + colMeanLum[Math.max(0, x - 5)]) * 0.5 - colMeanLum[x]);
      const score = colEdgeStrength[x] * 1.4 + lumDrop * 0.7 + localValley * 2.5 - colInk[x] * 4;
      if (score > maxRightSpineScore && score > 14) {
        maxRightSpineScore = score;
        bestRightBoundaryX = x;
      }
      if (x > mainTextRight + minMarginGapCols && colInk[x] >= 4 && colInk[x + 1] >= 4) {
        rightPageLimit = x - 2;
        break;
      }
    }
    rightPageLimit = Math.min(rightPageLimit, Math.max(bestRightBoundaryX, mainTextRight + Math.round(w * 0.02)));

    const mainRowInk = new Uint16Array(h);
    const rowLeftInk = new Int16Array(h).fill(mainTextLeft);
    const rowRightInk = new Int16Array(h).fill(mainTextRight);

    for (let y = Math.floor(h * 0.04); y < Math.floor(h * 0.96); y++) {
      let firstX = -1, lastX = -1, cnt = 0;
      for (let x = mainTextLeft; x <= mainTextRight; x++) {
        if (cleanInk[y * w + x]) {
          if (firstX === -1) firstX = x;
          lastX = x;
          cnt++;
        }
      }
      mainRowInk[y] = cnt;
      if (cnt >= 2) {
        rowLeftInk[y] = firstX;
        rowRightInk[y] = lastX;
      }
    }

    let mainTextTop = Math.floor(h * 0.10);
    let mainTextBottom = Math.floor(h * 0.90);
    const minMarginGapRows = Math.max(4, Math.round(h * 0.028));

    for (let y = Math.floor(h * 0.45); y >= Math.floor(h * 0.03); y--) {
      let emptyRun = 0;
      for (let k = 0; k < minMarginGapRows && y - k >= 0; k++) {
        if (mainRowInk[y - k] <= 1) emptyRun++;
      }
      if (emptyRun >= minMarginGapRows) {
        mainTextTop = y + 1;
        break;
      }
    }

    for (let y = Math.floor(h * 0.55); y <= Math.floor(h * 0.97); y++) {
      let emptyRun = 0;
      for (let k = 0; k < minMarginGapRows && y + k < h; k++) {
        if (mainRowInk[y + k] <= 1) emptyRun++;
      }
      if (emptyRun >= minMarginGapRows) {
        mainTextBottom = y - 1;
        break;
      }
    }

    // GIAI ĐOẠN 3: Quét Từ Trong Ra Ngoài (Inside-Out Edge Scan)
    const leftEdgePts = [];
    const rightEdgePts = [];
    const topEdgePts = [];
    const bottomEdgePts = [];

    const maxSearchMarginX = Math.round(w * 0.095);
    const defaultNicePadX = Math.round(w * 0.034);
    const minSafePadX = Math.round(w * 0.018);

    const scanY0 = Math.max(Math.floor(h * 0.12), mainTextTop);
    const scanY1 = Math.min(Math.floor(h * 0.88), mainTextBottom);

    for (let y = scanY0; y <= scanY1; y += 2) {
      const anchorLeftX = mainRowInk[y] >= 2 ? Math.min(mainTextLeft + Math.round(w * 0.04), rowLeftInk[y]) : mainTextLeft;
      const startLeftX = Math.max(3, anchorLeftX - minSafePadX);
      const stopLeftX = Math.max(3, Math.min(leftPageLimit, anchorLeftX - maxSearchMarginX));

      let foundLeftX = Math.max(leftPageLimit, anchorLeftX - defaultNicePadX);
      let bestLeftGrad = 14;

      for (let x = startLeftX; x >= stopLeftX; x--) {
        const idxIn = y * w + Math.min(w - 1, x + 2);
        const idxOut = y * w + Math.max(0, x - 2);
        const idxFarOut = y * w + Math.max(0, x - 4);

        const lumStep = closedLum[idxIn] - closedLum[idxOut];
        const colorStep = colorDist[idxOut] - colorDist[idxIn];
        const spineDip = (closedLum[idxIn] + closedLum[idxFarOut]) * 0.5 - closedLum[y * w + x];
        const gradScore = lumStep + colorStep * 0.85 + Math.max(0, spineDip * 2.2);

        if (gradScore > bestLeftGrad) {
          bestLeftGrad = gradScore;
          foundLeftX = x;
        }
        if (!isPaperSurface[idxOut] && !isPaperSurface[idxFarOut]) {
          if (bestLeftGrad <= 14) foundLeftX = x;
          break;
        }
      }
      leftEdgePts.push({ x: foundLeftX, y });

      const anchorRightX = mainRowInk[y] >= 2 ? Math.max(mainTextRight - Math.round(w * 0.04), rowRightInk[y]) : mainTextRight;
      const startRightX = Math.min(w - 4, anchorRightX + minSafePadX);
      const stopRightX = Math.min(w - 4, Math.max(rightPageLimit, anchorRightX + maxSearchMarginX));

      let foundRightX = Math.min(rightPageLimit, anchorRightX + defaultNicePadX);
      let bestRightGrad = 14;

      for (let x = startRightX; x <= stopRightX; x++) {
        const idxIn = y * w + Math.max(0, x - 2);
        const idxOut = y * w + Math.min(w - 1, x + 2);
        const idxFarOut = y * w + Math.min(w - 1, x + 4);

        const lumStep = closedLum[idxIn] - closedLum[idxOut];
        const colorStep = colorDist[idxOut] - colorDist[idxIn];
        const spineDip = (closedLum[idxIn] + closedLum[idxFarOut]) * 0.5 - closedLum[y * w + x];
        const gradScore = lumStep + colorStep * 0.85 + Math.max(0, spineDip * 2.2);

        if (gradScore > bestRightGrad) {
          bestRightGrad = gradScore;
          foundRightX = x;
        }
        if (!isPaperSurface[idxOut] && !isPaperSurface[idxFarOut]) {
          if (bestRightGrad <= 14) foundRightX = x;
          break;
        }
      }
      rightEdgePts.push({ x: foundRightX, y });
    }

    const maxSearchMarginY = Math.round(h * 0.10);
    const defaultNicePadY = Math.round(h * 0.034);
    const minSafePadY = Math.round(h * 0.018);

    const scanX0 = Math.max(Math.floor(w * 0.14), mainTextLeft);
    const scanX1 = Math.min(Math.floor(w * 0.86), mainTextRight);

    for (let x = scanX0; x <= scanX1; x += 2) {
      const startTopY = Math.max(3, mainTextTop - minSafePadY);
      const stopTopY = Math.max(3, mainTextTop - maxSearchMarginY);
      let foundTopY = Math.max(3, mainTextTop - defaultNicePadY);
      let bestTopGrad = 14;

      for (let y = startTopY; y >= stopTopY; y--) {
        const idxIn = Math.min(h - 1, y + 2) * w + x;
        const idxOut = Math.max(0, y - 2) * w + x;
        const idxFarOut = Math.max(0, y - 4) * w + x;

        const lumStep = closedLum[idxIn] - closedLum[idxOut];
        const colorStep = colorDist[idxOut] - colorDist[idxIn];
        const gradScore = lumStep + colorStep * 0.85;

        if (gradScore > bestTopGrad) {
          bestTopGrad = gradScore;
          foundTopY = y;
        }
        if (!isPaperSurface[idxOut] && !isPaperSurface[idxFarOut]) {
          if (bestTopGrad <= 14) foundTopY = y;
          break;
        }
      }
      topEdgePts.push({ x, y: foundTopY });

      const startBotY = Math.min(h - 4, mainTextBottom + minSafePadY);
      const stopBotY = Math.min(h - 4, mainTextBottom + maxSearchMarginY);
      let foundBotY = Math.min(h - 4, mainTextBottom + defaultNicePadY);
      let bestBotGrad = 14;

      for (let y = startBotY; y <= stopBotY; y++) {
        const idxIn = Math.max(0, y - 2) * w + x;
        const idxOut = Math.min(h - 1, y + 2) * w + x;
        const idxFarOut = Math.min(h - 1, y + 4) * w + x;

        const lumStep = closedLum[idxIn] - closedLum[idxOut];
        const colorStep = colorDist[idxOut] - colorDist[idxIn];
        const gradScore = lumStep + colorStep * 0.85;

        if (gradScore > bestBotGrad) {
          bestBotGrad = gradScore;
          foundBotY = y;
        }
        if (!isPaperSurface[idxOut] && !isPaperSurface[idxFarOut]) {
          if (bestBotGrad <= 14) foundBotY = y;
          break;
        }
      }
      bottomEdgePts.push({ x, y: foundBotY });
    }

    tempCanvas.width = 0;
    tempCanvas.height = 0;

    // GIAI ĐOẠN 4: Khớp 4 đường thẳng viền trang sách và tìm 4 góc giao điểm
    const leftLine = fitLineXY(leftEdgePts, 'x_of_y', Math.max(4, mainTextLeft - defaultNicePadX), h);
    const rightLine = fitLineXY(rightEdgePts, 'x_of_y', Math.min(w - 5, mainTextRight + defaultNicePadX), h);
    const topLine = fitLineXY(topEdgePts, 'y_of_x', Math.max(4, mainTextTop - defaultNicePadY), w);
    const bottomLine = fitLineXY(bottomEdgePts, 'y_of_x', Math.min(h - 5, mainTextBottom + defaultNicePadY), w);

    let tl = intersectLines(leftLine, topLine, w, h, { x: mainTextLeft - defaultNicePadX, y: mainTextTop - defaultNicePadY });
    let tr = intersectLines(rightLine, topLine, w, h, { x: mainTextRight + defaultNicePadX, y: mainTextTop - defaultNicePadY });
    let br = intersectLines(rightLine, bottomLine, w, h, { x: mainTextRight + defaultNicePadX, y: mainTextBottom + defaultNicePadY });
    let bl = intersectLines(leftLine, bottomLine, w, h, { x: mainTextLeft - defaultNicePadX, y: mainTextBottom + defaultNicePadY });

    const shrink = 0.005;
    const centerPt = {
      x: (tl.x + tr.x + br.x + bl.x) * 0.25,
      y: (tl.y + tr.y + br.y + bl.y) * 0.25
    };
    tl = { x: tl.x + (centerPt.x - tl.x) * shrink, y: tl.y + (centerPt.y - tl.y) * shrink };
    tr = { x: tr.x + (centerPt.x - tr.x) * shrink, y: tr.y + (centerPt.y - tr.y) * shrink };
    br = { x: br.x + (centerPt.x - br.x) * shrink, y: br.y + (centerPt.y - br.y) * shrink };
    bl = { x: bl.x + (centerPt.x - bl.x) * shrink, y: bl.y + (centerPt.y - bl.y) * shrink };

    const safeL = Math.max(2, mainTextLeft - minSafePadX);
    const safeR = Math.min(w - 3, mainTextRight + minSafePadX);
    const safeT = Math.max(2, mainTextTop - minSafePadY);
    const safeB = Math.min(h - 3, mainTextBottom + minSafePadY);

    tl.x = Math.min(tl.x, safeL);
    bl.x = Math.min(bl.x, safeL);
    tr.x = Math.max(tr.x, safeR);
    br.x = Math.max(br.x, safeR);

    tl.y = Math.min(tl.y, safeT);
    tr.y = Math.min(tr.y, safeT);
    bl.y = Math.max(bl.y, safeB);
    br.y = Math.max(br.y, safeB);

    const quad = {
      tl: { x: clamp(tl.x / w, 0.01, 0.46), y: clamp(tl.y / h, 0.01, 0.46) },
      tr: { x: clamp(tr.x / w, 0.54, 0.99), y: clamp(tr.y / h, 0.01, 0.46) },
      br: { x: clamp(br.x / w, 0.54, 0.99), y: clamp(br.y / h, 0.54, 0.99) },
      bl: { x: clamp(bl.x / w, 0.01, 0.46), y: clamp(bl.y / h, 0.54, 0.99) }
    };

    if (computeQuadNormalizedArea(quad) < 0.20) return getDefaultInsetQuad();
    return quad;
  }

  function fitLineXY(pts, mode, fallbackConst, span) {
    if (pts.length < 6) {
      return { m: 0, c: fallbackConst };
    }
    const half = Math.floor(pts.length / 2);
    const firstHalf = pts.slice(0, half);
    const secondHalf = pts.slice(half);

    const getMedianPt = (arr) => {
      const xs = arr.map((p) => p.x).sort((a, b) => a - b);
      const ys = arr.map((p) => p.y).sort((a, b) => a - b);
      const mid = Math.floor(arr.length / 2);
      return { x: xs[mid], y: ys[mid] };
    };

    const p1 = getMedianPt(firstHalf);
    const p2 = getMedianPt(secondHalf);

    if (mode === 'x_of_y') {
      const dy = p2.y - p1.y;
      if (Math.abs(dy) < span * 0.1) return { m: 0, c: (p1.x + p2.x) * 0.5 };
      const m = clamp((p2.x - p1.x) / dy, -0.35, 0.35);
      const c = p1.x - m * p1.y;
      return { m, c };
    } else {
      const dx = p2.x - p1.x;
      if (Math.abs(dx) < span * 0.1) return { m: 0, c: (p1.y + p2.y) * 0.5 };
      const m = clamp((p2.y - p1.y) / dx, -0.35, 0.35);
      const c = p1.y - m * p1.x;
      return { m, c };
    }
  }

  function intersectLines(vertLine, horizLine, w, h, fallback) {
    const denom = 1 - vertLine.m * horizLine.m;
    if (Math.abs(denom) < 1e-5) return fallback;
    const x = (vertLine.m * horizLine.c + vertLine.c) / denom;
    const y = horizLine.m * x + horizLine.c;
    return {
      x: clamp(x, 1, w - 2),
      y: clamp(y, 1, h - 2)
    };
  }

  /**
   * Giải hệ 8 phương trình tuyến tính Homography bằng khử Gauss-Jordan
   */
  function computeHomographyMatrix(srcPts, dstPts) {
    const A = [];
    const B = [];
    for (let i = 0; i < 4; i++) {
      const u = srcPts[i].x, v = srcPts[i].y;
      const x = dstPts[i].x, y = dstPts[i].y;
      A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]);
      B.push(x);
      A.push([0, 0, 0, u, v, 1, -u * y, -v * y]);
      B.push(y);
    }

    const n = 8;
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      let maxVal = Math.abs(A[col][col]);
      for (let row = col + 1; row < n; row++) {
        const absVal = Math.abs(A[row][col]);
        if (absVal > maxVal) { maxVal = absVal; maxRow = row; }
      }
      if (maxRow !== col) {
        const tmpRow = A[col]; A[col] = A[maxRow]; A[maxRow] = tmpRow;
        const tmpB = B[col]; B[col] = B[maxRow]; B[maxRow] = tmpB;
      }
      const pivot = A[col][col] || 1e-12;
      for (let j = col; j < n; j++) A[col][j] /= pivot;
      B[col] /= pivot;

      for (let row = 0; row < n; row++) {
        if (row !== col) {
          const factor = A[row][col];
          if (Math.abs(factor) > 1e-12) {
            for (let j = col; j < n; j++) A[row][j] -= factor * A[col][j];
            B[row] -= factor * B[col];
          }
        }
      }
    }
    return B;
  }

  /**
   * MODULE 2.2: Nắn Phẳng Phối Cảnh 4 Góc (4-Point Perspective Homography Warp)
   */
  function perspectiveWarp4Point(sourceImgOrCanvas, quad) {
    const srcW = sourceImgOrCanvas.naturalWidth || sourceImgOrCanvas.width;
    const srcH = sourceImgOrCanvas.naturalHeight || sourceImgOrCanvas.height;

    const pTL = { x: quad.tl.x * srcW, y: quad.tl.y * srcH };
    const pTR = { x: quad.tr.x * srcW, y: quad.tr.y * srcH };
    const pBR = { x: quad.br.x * srcW, y: quad.br.y * srcH };
    const pBL = { x: quad.bl.x * srcW, y: quad.bl.y * srcH };

    const widthTop = Math.hypot(pTR.x - pTL.x, pTR.y - pTL.y);
    const widthBottom = Math.hypot(pBR.x - pBL.x, pBR.y - pBL.y);
    const heightLeft = Math.hypot(pBL.x - pTL.x, pBL.y - pTL.y);
    const heightRight = Math.hypot(pBR.x - pTR.x, pBR.y - pTR.y);

    const dstW = Math.max(60, Math.round((widthTop + widthBottom) * 0.5));
    const dstH = Math.max(60, Math.round((heightLeft + heightRight) * 0.5));

    const isAxisAlignedRect =
      Math.abs(quad.tl.y - quad.tr.y) < 0.002 &&
      Math.abs(quad.bl.y - quad.br.y) < 0.002 &&
      Math.abs(quad.tl.x - quad.bl.x) < 0.002 &&
      Math.abs(quad.tr.x - quad.br.x) < 0.002;

    const outCanvas = document.createElement('canvas');
    outCanvas.width = dstW;
    outCanvas.height = dstH;
    const outCtx = outCanvas.getContext('2d');

    if (isAxisAlignedRect) {
      outCtx.imageSmoothingEnabled = true;
      outCtx.imageSmoothingQuality = 'high';
      outCtx.drawImage(
        sourceImgOrCanvas,
        pTL.x,
        pTL.y,
        Math.max(1, pTR.x - pTL.x),
        Math.max(1, pBL.y - pTL.y),
        0,
        0,
        dstW,
        dstH
      );
      return outCanvas;
    }

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.drawImage(sourceImgOrCanvas, 0, 0, srcW, srcH);
    const srcData = srcCtx.getImageData(0, 0, srcW, srcH).data;

    const H = computeHomographyMatrix(
      [
        { x: 0, y: 0 },
        { x: dstW - 1, y: 0 },
        { x: dstW - 1, y: dstH - 1 },
        { x: 0, y: dstH - 1 }
      ],
      [pTL, pTR, pBR, pBL]
    );

    const dstImageData = outCtx.createImageData(dstW, dstH);
    const dstPixels = dstImageData.data;

    const h00 = H[0], h01 = H[1], h02 = H[2];
    const h10 = H[3], h11 = H[4], h12 = H[5];
    const h20 = H[6], h21 = H[7], h22 = 1.0;

    const maxSrcX = srcW - 1.001;
    const maxSrcY = srcH - 1.001;

    let dstIdx = 0;
    for (let v = 0; v < dstH; v++) {
      const rowX = h01 * v + h02;
      const rowY = h11 * v + h12;
      const rowW = h21 * v + h22;

      for (let u = 0; u < dstW; u++) {
        const denom = h20 * u + rowW;
        const sx = clamp((h00 * u + rowX) / denom, 0, maxSrcX);
        const sy = clamp((h10 * u + rowY) / denom, 0, maxSrcY);

        const x0 = sx | 0;
        const y0 = sy | 0;
        const dx = sx - x0;
        const dy = sy - y0;
        const w00 = (1 - dx) * (1 - dy);
        const w10 = dx * (1 - dy);
        const w01 = (1 - dx) * dy;
        const w11 = dx * dy;

        const idx00 = (y0 * srcW + x0) << 2;
        const idx10 = idx00 + 4;
        const idx01 = idx00 + (srcW << 2);
        const idx11 = idx01 + 4;

        dstPixels[dstIdx]     = srcData[idx00] * w00 + srcData[idx10] * w10 + srcData[idx01] * w01 + srcData[idx11] * w11;
        dstPixels[dstIdx + 1] = srcData[idx00 + 1] * w00 + srcData[idx10 + 1] * w10 + srcData[idx01 + 1] * w01 + srcData[idx11 + 1] * w11;
        dstPixels[dstIdx + 2] = srcData[idx00 + 2] * w00 + srcData[idx10 + 2] * w10 + srcData[idx01 + 2] * w01 + srcData[idx11 + 2] * w11;
        dstPixels[dstIdx + 3] = 255;
        dstIdx += 4;
      }
    }

    outCtx.putImageData(dstImageData, 0, 0);
    srcCanvas.width = 0;
    srcCanvas.height = 0;
    return outCanvas;
  }

  /**
   * MODULE 3: Xoay 90°/180°/270° và Xoay Vi Chỉnh (-15° đến +15°)
   */
  function rotateAndStraightenCanvas(sourceCanvas, rotation90, fineAngleDeg) {
    const normRot = ((rotation90 % 360) + 360) % 360;
    if (normRot === 0 && Math.abs(fineAngleDeg) < 0.05) {
      return sourceCanvas;
    }

    const srcW = sourceCanvas.width;
    const srcH = sourceCanvas.height;
    const rotW = normRot === 90 || normRot === 270 ? srcH : srcW;
    const rotH = normRot === 90 || normRot === 270 ? srcW : srcH;

    const outCanvas = document.createElement('canvas');
    outCanvas.width = rotW;
    outCanvas.height = rotH;
    const ctx = outCanvas.getContext('2d');

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, rotW, rotH);

    ctx.save();
    ctx.translate(rotW / 2, rotH / 2);
    const totalRad = ((normRot + fineAngleDeg) * Math.PI) / 180;
    ctx.rotate(totalRad);

    const fineRad = Math.abs((fineAngleDeg * Math.PI) / 180);
    const autoZoom = fineRad > 0.001 ? Math.min(1.08, 1 + Math.sin(fineRad) * 0.45) : 1.0;
    ctx.scale(autoZoom, autoZoom);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(sourceCanvas, -srcW / 2, -srcH / 2, srcW, srcH);
    ctx.restore();

    return outCanvas;
  }

  function enhanceDocumentChannel(val, darknessGain) {
    if (val >= 224) {
      return 224 + (val - 224) * 1.45;
    }
    const diffFromWhite = 224 - val;
    return 224 - Math.pow(diffFromWhite / 224, 0.88) * 224 * darknessGain;
  }

  /**
   * MODULE 4: Bộ Lọc Làm Sạch Trang Sách Học Tập (Study Document Enhancer)
   */
  function applyStudyFilter(sourceCanvas, filterMode, brightness = 65, contrast = 65) {
    const w = sourceCanvas.width;
    const h = sourceCanvas.height;

    if (filterMode === 'original' && Math.abs(brightness - 50) <= 3 && Math.abs(contrast - 50) <= 3) {
      return sourceCanvas;
    }

    const outCanvas = document.createElement('canvas');
    outCanvas.width = w;
    outCanvas.height = h;
    const ctx = outCanvas.getContext('2d');
    ctx.drawImage(sourceCanvas, 0, 0);
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    if (filterMode === 'original') {
      const bOffset = (brightness - 50) * 1.4;
      const cFactor = Math.pow((contrast + 50) / 100, 1.5);
      for (let i = 0; i < data.length; i += 4) {
        data[i]     = clampByte((data[i] - 128) * cFactor + 128 + bOffset);
        data[i + 1] = clampByte((data[i + 1] - 128) * cFactor + 128 + bOffset);
        data[i + 2] = clampByte((data[i + 2] - 128) * cFactor + 128 + bOffset);
      }
      ctx.putImageData(imgData, 0, 0);
      return outCanvas;
    }

    const gridScale = Math.max(8, Math.round(Math.max(w, h) / 96));
    const gw = Math.max(8, Math.ceil(w / gridScale));
    const gh = Math.max(8, Math.ceil(h / gridScale));
    const bgLumGrid = new Float32Array(gw * gh);

    for (let gy = 0; gy < gh; gy++) {
      const y0 = gy * gridScale;
      const y1 = Math.min(h, y0 + gridScale);
      const stepY = Math.max(1, Math.floor((y1 - y0) / 3));
      for (let gx = 0; gx < gw; gx++) {
        const x0 = gx * gridScale;
        const x1 = Math.min(w, x0 + gridScale);
        const stepX = Math.max(1, Math.floor((x1 - x0) / 3));
        let maxL = 60;
        for (let y = y0; y < y1; y += stepY) {
          for (let x = x0; x < x1; x += stepX) {
            const idx = (y * w + x) << 2;
            const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            if (lum > maxL) maxL = lum;
          }
        }
        bgLumGrid[gy * gw + gx] = maxL;
      }
    }

    let blurredBg = bgLumGrid;
    for (let pass = 0; pass < 3; pass++) {
      const nextGrid = new Float32Array(gw * gh);
      for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
          let sum = 0, cnt = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const ny = clamp(gy + dy, 0, gh - 1);
            for (let dx = -1; dx <= 1; dx++) {
              const nx = clamp(gx + dx, 0, gw - 1);
              sum += blurredBg[ny * gw + nx];
              cnt++;
            }
          }
          nextGrid[gy * gw + gx] = sum / cnt;
        }
      }
      blurredBg = nextGrid;
    }

    const paperWhiteBoost = 242 + (brightness - 50) * 0.45;
    const inkDarknessGain = 1.0 + (contrast - 50) * 0.018;

    for (let y = 0; y < h; y++) {
      const gy = y / gridScale;
      const gy0 = clamp(Math.floor(gy), 0, gh - 1);
      const gy1 = clamp(gy0 + 1, 0, gh - 1);
      const fy = gy - gy0;

      for (let x = 0; x < w; x++) {
        const gx = x / gridScale;
        const gx0 = clamp(Math.floor(gx), 0, gw - 1);
        const gx1 = clamp(gx0 + 1, 0, gw - 1);
        const fx = gx - gx0;

        const b00 = blurredBg[gy0 * gw + gx0];
        const b10 = blurredBg[gy0 * gw + gx1];
        const b01 = blurredBg[gy1 * gw + gx0];
        const b11 = blurredBg[gy1 * gw + gx1];
        const localBg = Math.max(
          55,
          b00 * (1 - fx) * (1 - fy) +
          b10 * fx * (1 - fy) +
          b01 * (1 - fx) * fy +
          b11 * fx * fy
        );

        const idx = (y * w + x) << 2;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];

        if (filterMode === 'magic_color') {
          const maxC = Math.max(r, g, b);
          const minC = Math.min(r, g, b);
          const chroma = maxC - minC;

          let nr = (r / localBg) * paperWhiteBoost;
          let ng = (g / localBg) * paperWhiteBoost;
          let nb = (b / localBg) * paperWhiteBoost;

          if (chroma > 28) {
            const avg = (nr + ng + nb) / 3;
            const satBoost = 1.28;
            nr = avg + (nr - avg) * satBoost;
            ng = avg + (ng - avg) * satBoost;
            nb = avg + (nb - avg) * satBoost;
          } else {
            nr = enhanceDocumentChannel(nr, inkDarknessGain);
            ng = enhanceDocumentChannel(ng, inkDarknessGain);
            nb = enhanceDocumentChannel(nb, inkDarknessGain);
          }

          data[idx]     = clampByte(nr);
          data[idx + 1] = clampByte(ng);
          data[idx + 2] = clampByte(nb);
        } else if (filterMode === 'bw_sharp') {
          const pixelLum = 0.299 * r + 0.587 * g + 0.114 * b;
          const normLum = (pixelLum / localBg) * paperWhiteBoost;
          const thresholdCenter = 202 - (contrast - 50) * 0.65;
          const steepness = 0.14 * inkDarknessGain;
          const sig = 255 / (1 + Math.exp(-(normLum - thresholdCenter) * steepness));
          const bwVal = sig > 238 ? 255 : sig < 18 ? 0 : sig;

          data[idx] = bwVal;
          data[idx + 1] = bwVal;
          data[idx + 2] = bwVal;
        } else if (filterMode === 'grayscale') {
          const pixelLum = 0.299 * r + 0.587 * g + 0.114 * b;
          const normLum = (pixelLum / localBg) * (paperWhiteBoost - 6);
          const grayVal = clampByte(enhanceDocumentChannel(normLum, inkDarknessGain * 0.85));

          data[idx] = grayVal;
          data[idx + 1] = grayVal;
          data[idx + 2] = grayVal;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
    return outCanvas;
  }

  async function processPagePipeline(page) {
    const img = await loadImageElement(page.originalDataUrl);
    const warpedCanvas = perspectiveWarp4Point(img, page.cropQuad);
    const rotatedCanvas = rotateAndStraightenCanvas(warpedCanvas, page.rotation90, page.fineAngle);
    if (rotatedCanvas !== warpedCanvas) {
      warpedCanvas.width = 0;
      warpedCanvas.height = 0;
    }
    const filteredCanvas = applyStudyFilter(
      rotatedCanvas,
      page.filterMode,
      page.brightness,
      page.contrast
    );
    if (filteredCanvas !== rotatedCanvas) {
      rotatedCanvas.width = 0;
      rotatedCanvas.height = 0;
    }

    const width = filteredCanvas.width;
    const height = filteredCanvas.height;
    const processedDataUrl = filteredCanvas.toDataURL('image/jpeg', 0.9);
    filteredCanvas.width = 0;
    filteredCanvas.height = 0;

    return { processedDataUrl, width, height };
  }

  /**
   * Tạo 2 trang sách giáo trình mẫu chụp nghiêng trên mặt bàn gỗ có bóng râm & highlight
   */
  async function generateDemoStudyPages() {
    const demoConfigs = [
      {
        pageNum: 1,
        title: 'CHƯƠNG 1: SINH LÝ BỆNH SUY TIM MẠN TÍNH',
        subtitle: 'Giáo trình Nội Khoa Cơ Sở - Đại Học Y Dược',
        quad: {
          tl: { x: 0.11, y: 0.08 },
          tr: { x: 0.89, y: 0.11 },
          br: { x: 0.93, y: 0.91 },
          bl: { x: 0.08, y: 0.89 }
        },
        highlights: [
          { line: 1, color: 'rgba(250, 204, 21, 0.58)' },
          { line: 4, color: 'rgba(244, 114, 182, 0.45)' }
        ],
        bullets: [
          '1. Định nghĩa: Hội chứng lâm sàng phức tạp do bất thường cấu trúc/chức năng tim.',
          '2. Cơ chế bù trừ thần kinh - thể dịch: Hoạt hóa hệ giao cảm & hệ RAA.',
          '3. Hậu quả kéo dài: Tái cấu trúc cơ tim (Remodeling) và giảm phân suất tống máu.',
          '4. Phân loại theo EF: HFrEF (EF <= 40%), HFmrEF (41-49%), HFpEF (>= 50%).',
          '5. Triệu chứng cơ năng chính: Khó thở khi gắng sức, orthopnea, phù 2 chi dưới.',
          '6. Dấu ấn sinh học chuẩn vàng: BNP >= 35 pg/mL hoặc NT-proBNP >= 125 pg/mL.'
        ],
        diagramType: 'ecg'
      },
      {
        pageNum: 2,
        title: 'PHÁC ĐỒ TỨ TRỤ ĐIỀU TRỊ HFrEF (ESC/AHA)',
        subtitle: 'Tài liệu Ôn Tập Nội Trú - Trang 02',
        quad: {
          tl: { x: 0.09, y: 0.10 },
          tr: { x: 0.91, y: 0.07 },
          br: { x: 0.88, y: 0.92 },
          bl: { x: 0.12, y: 0.90 }
        },
        highlights: [
          { line: 0, color: 'rgba(250, 204, 21, 0.58)' },
          { line: 3, color: 'rgba(56, 189, 248, 0.45)' }
        ],
        bullets: [
          '• Nhóm 1: ARNI (Sacubitril/Valsartan) hoặc ACEi/ARB giảm tỷ lệ tử vong.',
          '• Nhóm 2: Chẹn Beta giao cảm (Bisoprolol, Metoprolol succinate, Carvedilol).',
          '• Nhóm 3: Kháng Aldosterone - MRA (Spironolactone / Eplerenone).',
          '• Nhóm 4: Ức chế SGLT2 (Dapagliflozin / Empagliflozin) bất kể đái tháo đường.',
          '• Lợi tiểu quai (Furosemide): Kiểm soát triệu chứng sung huyết & ứ dịch.',
          '• Lưu ý lâm sàng: Theo dõi eGFR và Kali máu sau 1-2 tuần khởi trị.'
        ],
        diagramType: 'chart'
      }
    ];

    const results = [];
    for (const cfg of demoConfigs) {
      const w = 1200, h = 1600;
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');

      const deskGrad = ctx.createLinearGradient(0, 0, w, h);
      deskGrad.addColorStop(0, '#2c1e16');
      deskGrad.addColorStop(0.5, '#3b281d');
      deskGrad.addColorStop(1, '#1e140e');
      ctx.fillStyle = deskGrad;
      ctx.fillRect(0, 0, w, h);

      const pageW = 940, pageH = 1320;
      const pageCanvas = document.createElement('canvas');
      pageCanvas.width = pageW;
      pageCanvas.height = pageH;
      const pCtx = pageCanvas.getContext('2d');

      pCtx.fillStyle = '#f6f3e8';
      pCtx.fillRect(0, 0, pageW, pageH);

      pCtx.fillStyle = '#1e3a8a';
      pCtx.font = 'bold 31px -apple-system, BlinkMacSystemFont, sans-serif';
      pCtx.fillText(cfg.title, 65, 95);

      pCtx.fillStyle = '#64748b';
      pCtx.font = 'italic 22px -apple-system, BlinkMacSystemFont, sans-serif';
      pCtx.fillText(cfg.subtitle, 65, 132);

      let curY = 215;
      cfg.bullets.forEach((text, idx) => {
        const hl = cfg.highlights.find((hItem) => hItem.line === idx);
        if (hl) {
          pCtx.fillStyle = hl.color;
          pCtx.fillRect(60, curY - 30, pageW - 120, 44);
        }
        pCtx.fillStyle = '#1e293b';
        pCtx.font = '500 24px -apple-system, BlinkMacSystemFont, sans-serif';
        pCtx.fillText(text, 68, curY, pageW - 136);
        curY += 76;
      });

      pCtx.fillStyle = '#eff6ff';
      pCtx.strokeStyle = '#3b82f6';
      pCtx.lineWidth = 3;
      pCtx.fillRect(65, 700, pageW - 130, 390);
      pCtx.strokeRect(65, 700, pageW - 130, 390);

      pCtx.fillStyle = '#1d4ed8';
      pCtx.font = 'bold 24px sans-serif';
      pCtx.fillText(
        cfg.diagramType === 'ecg'
          ? 'SƠ ĐỒ 1.1: ĐƯỜNG CONG FRANK-STARLING & ÁP LỰC ĐỔ ĐẦY'
          : 'SƠ ĐỒ 1.2: HIỆU QUẢ GIẢM TỬ VONG KHI PHỐI HỢP TỨ TRỤ SỚM',
        95,
        750
      );

      if (cfg.diagramType === 'ecg') {
        pCtx.strokeStyle = '#16a34a';
        pCtx.lineWidth = 5;
        pCtx.beginPath();
        pCtx.moveTo(120, 1020);
        pCtx.quadraticCurveTo(360, 770, 780, 810);
        pCtx.stroke();

        pCtx.strokeStyle = '#dc2626';
        pCtx.lineWidth = 5;
        pCtx.beginPath();
        pCtx.moveTo(120, 1020);
        pCtx.quadraticCurveTo(360, 910, 780, 955);
        pCtx.stroke();
      } else {
        const bars = [
          { label: 'ARNI', val: 260, color: '#2563eb' },
          { label: 'Beta-Blocker', val: 230, color: '#0891b2' },
          { label: 'MRA', val: 200, color: '#16a34a' },
          { label: 'SGLT2i', val: 245, color: '#f59e0b' }
        ];
        bars.forEach((b, i) => {
          const bx = 150 + i * 180;
          const by = 1030 - b.val;
          pCtx.fillStyle = b.color;
          pCtx.fillRect(bx, by, 95, b.val);
          pCtx.fillStyle = '#0f172a';
          pCtx.font = 'bold 20px sans-serif';
          pCtx.fillText(b.label, bx + 6, 1062);
        });
      }

      const shadowGrad = pCtx.createLinearGradient(0, 0, pageW, pageH);
      shadowGrad.addColorStop(0, 'rgba(0, 0, 0, 0.24)');
      shadowGrad.addColorStop(0.3, 'rgba(0, 0, 0, 0.05)');
      shadowGrad.addColorStop(1, 'rgba(15, 23, 42, 0.28)');
      pCtx.fillStyle = shadowGrad;
      pCtx.fillRect(0, 0, pageW, pageH);

      const srcPageData = pCtx.getImageData(0, 0, pageW, pageH).data;
      const deskImgData = ctx.getImageData(0, 0, w, h);
      const deskPixels = deskImgData.data;

      const H = computeHomographyMatrix(
        [
          { x: cfg.quad.tl.x * w, y: cfg.quad.tl.y * h },
          { x: cfg.quad.tr.x * w, y: cfg.quad.tr.y * h },
          { x: cfg.quad.br.x * w, y: cfg.quad.br.y * h },
          { x: cfg.quad.bl.x * w, y: cfg.quad.bl.y * h }
        ],
        [
          { x: 0, y: 0 },
          { x: pageW - 1, y: 0 },
          { x: pageW - 1, y: pageH - 1 },
          { x: 0, y: pageH - 1 }
        ]
      );

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const denom = H[6] * x + H[7] * y + 1;
          const px = (H[0] * x + H[1] * y + H[2]) / denom;
          const py = (H[3] * x + H[4] * y + H[5]) / denom;
          if (px >= 0 && px < pageW - 1 && py >= 0 && py < pageH - 1) {
            const sx = px | 0;
            const sy = py | 0;
            const sIdx = (sy * pageW + sx) << 2;
            const dIdx = (y * w + x) << 2;
            deskPixels[dIdx]     = srcPageData[sIdx];
            deskPixels[dIdx + 1] = srcPageData[sIdx + 1];
            deskPixels[dIdx + 2] = srcPageData[sIdx + 2];
          }
        }
      }

      ctx.putImageData(deskImgData, 0, 0);
      results.push(canvas.toDataURL('image/jpeg', 0.9));
      pageCanvas.width = 0;
      canvas.width = 0;
    }

    return results;
  }

  /**
   * IndexedDB Storage Engine
   */
  const DB_NAME = 'DocScanStudyPDF_DB';
  const DB_VERSION = 1;
  const STORE_ACTIVE_PAGES = 'active_pages';
  const STORE_SAVED_PDFS = 'saved_pdfs';

  function openDatabase() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_ACTIVE_PAGES)) {
          db.createObjectStore(STORE_ACTIVE_PAGES, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORE_SAVED_PDFS)) {
          db.createObjectStore(STORE_SAVED_PDFS, { keyPath: 'id' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async function saveActiveWorkspacePages(pages) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_ACTIVE_PAGES, 'readwrite');
      const store = tx.objectStore(STORE_ACTIVE_PAGES);
      store.clear();
      pages.forEach((page, orderIndex) => {
        store.put({ ...page, orderIndex });
      });
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  async function loadActiveWorkspacePages() {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_ACTIVE_PAGES, 'readonly');
      const store = tx.objectStore(STORE_ACTIVE_PAGES);
      const req = store.getAll();
      req.onsuccess = () => {
        const items = req.result || [];
        items.sort((a, b) => (a.orderIndex ?? 0) - (b.orderIndex ?? 0));
        db.close();
        resolve(items);
      };
      req.onerror = () => { db.close(); reject(req.error); };
    });
  }

  async function savePdfToAppLibrary(record) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SAVED_PDFS, 'readwrite');
      tx.objectStore(STORE_SAVED_PDFS).put(record);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  async function getAllSavedPdfs() {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SAVED_PDFS, 'readonly');
      const req = tx.objectStore(STORE_SAVED_PDFS).getAll();
      req.onsuccess = () => {
        const list = req.result || [];
        list.sort((a, b) => b.createdAt - a.createdAt);
        db.close();
        resolve(list);
      };
      req.onerror = () => { db.close(); reject(req.error); };
    });
  }

  async function renameSavedPdfInLibrary(id, newFileName) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SAVED_PDFS, 'readwrite');
      const store = tx.objectStore(STORE_SAVED_PDFS);
      const getReq = store.get(id);
      getReq.onsuccess = () => {
        const item = getReq.result;
        if (item) {
          item.fileName = sanitizePdfFileName(newFileName);
          store.put(item);
        }
      };
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  async function deleteSavedPdfFromLibrary(id) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SAVED_PDFS, 'readwrite');
      tx.objectStore(STORE_SAVED_PDFS).delete(id);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  function sanitizePdfFileName(rawName) {
    const trimmed = (rawName || '').trim().replace(/[\\/:*?"<>|]+/g, '_') || 'Tai_Lieu_DocScan';
    return trimmed.toLowerCase().endsWith('.pdf') ? trimmed : `${trimmed}.pdf`;
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function base64ToBlob(base64DataUrl, mimeType = 'application/pdf') {
    const parts = base64DataUrl.split(',');
    const base64 = parts.length > 1 ? parts[1] : parts[0];
    const byteStr = atob(base64);
    const ab = new ArrayBuffer(byteStr.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteStr.length; i++) {
      ia[i] = byteStr.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeType });
  }

  /**
   * MODULE 5: Multi-page PDF Generator (jsPDF) + iOS Share Sheet
   */
  async function generateMultiPageStudyPdf(pages, options, onProgress) {
    const jsPDFClass = global.jspdf ? global.jspdf.jsPDF : null;
    if (!jsPDFClass) throw new Error('Chưa nạp được thư viện jsPDF.');

    const finalFileName = sanitizePdfFileName(options.fileName);
    const qualityConfig = {
      hd: { maxSide: 2200, jpegQuality: 0.92 },
      standard: { maxSide: 1750, jpegQuality: 0.82 },
      light: { maxSide: 1300, jpegQuality: 0.68 }
    }[options.quality || 'hd'];

    let pdf = null;
    const exportedPageImages = [];

    for (let i = 0; i < pages.length; i++) {
      if (onProgress) onProgress(i + 1, pages.length);
      const page = pages[i];
      const img = await loadImageElement(page.processedDataUrl);
      const rawW = img.naturalWidth || img.width;
      const rawH = img.naturalHeight || img.height;

      const scale = Math.min(1, qualityConfig.maxSide / Math.max(rawW, rawH));
      const targetW = Math.max(100, Math.round(rawW * scale));
      const targetH = Math.max(100, Math.round(rawH * scale));

      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, targetW, targetH);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, targetW, targetH);

      const compressedDataUrl = canvas.toDataURL('image/jpeg', qualityConfig.jpegQuality);
      exportedPageImages.push(compressedDataUrl);
      canvas.width = 0;
      canvas.height = 0;

      const effectiveMode = options.orientation !== 'auto' ? options.orientation : page.orientation;
      let isLandscape = targetW > targetH;
      if (effectiveMode === 'portrait') isLandscape = false;
      else if (effectiveMode === 'landscape') isLandscape = true;

      const orientationCode = isLandscape ? 'l' : 'p';

      if (options.layout === 'fit') {
        let pageMmW = targetW;
        let pageMmH = targetH;
        if (effectiveMode === 'portrait' && targetW > targetH) {
          pageMmW = 210; pageMmH = 297;
        } else if (effectiveMode === 'landscape' && targetH > targetW) {
          pageMmW = 297; pageMmH = 210;
        } else {
          const baseMm = 210;
          if (targetW <= targetH) {
            pageMmW = baseMm;
            pageMmH = Math.round((targetH / targetW) * baseMm * 100) / 100;
          } else {
            pageMmH = baseMm;
            pageMmW = Math.round((targetW / targetH) * baseMm * 100) / 100;
          }
        }

        const formatArray = [Math.min(pageMmW, pageMmH), Math.max(pageMmW, pageMmH)];
        if (i === 0 || !pdf) {
          pdf = new jsPDFClass({
            orientation: pageMmW > pageMmH ? 'l' : 'p',
            unit: 'mm',
            format: formatArray,
            compress: true
          });
        } else {
          pdf.addPage(formatArray, pageMmW > pageMmH ? 'l' : 'p');
        }

        const imgRatio = targetW / targetH;
        const pageRatio = pageMmW / pageMmH;
        let drawW = pageMmW, drawH = pageMmH, offsetX = 0, offsetY = 0;
        if (Math.abs(imgRatio - pageRatio) > 0.01) {
          if (imgRatio > pageRatio) {
            drawW = pageMmW; drawH = pageMmW / imgRatio; offsetY = (pageMmH - drawH) / 2;
          } else {
            drawH = pageMmH; drawW = pageMmH * imgRatio; offsetX = (pageMmW - drawW) / 2;
          }
        }
        pdf.addImage(compressedDataUrl, 'JPEG', offsetX, offsetY, drawW, drawH, undefined, 'FAST');
      } else {
        const pageMmW = isLandscape ? 297 : 210;
        const pageMmH = isLandscape ? 210 : 297;
        const marginMm = 7;
        const availW = pageMmW - marginMm * 2;
        const availH = pageMmH - marginMm * 2;

        if (i === 0 || !pdf) {
          pdf = new jsPDFClass({ orientation: orientationCode, unit: 'mm', format: 'a4', compress: true });
        } else {
          pdf.addPage('a4', orientationCode);
        }

        const imgRatio = targetW / targetH;
        const availRatio = availW / availH;
        let drawW = availW, drawH = availH;
        if (imgRatio > availRatio) {
          drawW = availW; drawH = availW / imgRatio;
        } else {
          drawH = availH; drawW = availH * imgRatio;
        }
        const offsetX = marginMm + (availW - drawW) / 2;
        const offsetY = marginMm + (availH - drawH) / 2;
        pdf.addImage(compressedDataUrl, 'JPEG', offsetX, offsetY, drawW, drawH, undefined, 'FAST');
      }
    }

    const pdfArrayBuffer = pdf.output('arraybuffer');
    const pdfBlob = new Blob([pdfArrayBuffer], { type: 'application/pdf' });
    const pdfBase64 = await blobToBase64(pdfBlob);

    return {
      fileName: finalFileName,
      pdfBlob,
      pdfBase64,
      fileSizeBytes: pdfBlob.size,
      pageCount: pages.length,
      pageImages: exportedPageImages
    };
  }

  async function sharePdfViaIOSShareSheet(pdfBlob, fileName) {
    const cleanName = sanitizePdfFileName(fileName);
    const pdfFile = new File([pdfBlob], cleanName, {
      type: 'application/pdf',
      lastModified: Date.now()
    });

    if (typeof navigator.share === 'function') {
      const shareData = {
        files: [pdfFile],
        title: cleanName,
        text: `Tài liệu học tập: ${cleanName} (DocScan Study PDF)`
      };
      if (!navigator.canShare || navigator.canShare({ files: [pdfFile] })) {
        try {
          await navigator.share(shareData);
          return { shared: true, message: 'Đã mở bảng Lưu vào Tệp / Chia sẻ của iOS!' };
        } catch (err) {
          if (err && err.name === 'AbortError') {
            return { shared: false, message: 'Đã đóng menu Chia sẻ.' };
          }
        }
      }
    }

    triggerDirectDownloadPdf(pdfBlob, cleanName);
    return {
      shared: false,
      message: 'Trình duyệt trên máy tính không có iOS Share Sheet nên đã tự động Tải xuống file PDF!'
    };
  }

  function triggerDirectDownloadPdf(pdfBlob, fileName) {
    const cleanName = sanitizePdfFileName(fileName);
    const url = URL.createObjectURL(pdfBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = cleanName;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 2000);
  }

  global.DocScanEngine = {
    loadImageElement,
    getDefaultFullQuad,
    compressForIOSMemoryGuard,
    detectDocumentCorners,
    perspectiveWarp4Point,
    rotateAndStraightenCanvas,
    applyStudyFilter,
    processPagePipeline,
    generateDemoStudyPages,
    saveActiveWorkspacePages,
    loadActiveWorkspacePages,
    savePdfToAppLibrary,
    getAllSavedPdfs,
    renameSavedPdfInLibrary,
    deleteSavedPdfFromLibrary,
    sanitizePdfFileName,
    base64ToBlob,
    generateMultiPageStudyPdf,
    sharePdfViaIOSShareSheet,
    triggerDirectDownloadPdf
  };
})(window);
