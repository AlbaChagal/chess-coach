export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function nextClockwiseCornerIndex(cornerIndex) {
  if (cornerIndex === 0) {
    return 1;
  }
  if (cornerIndex === 1) {
    return 2;
  }
  if (cornerIndex === 2) {
    return 3;
  }
  return 0;
}

export function getContainedImageBounds(layout, imageWidth, imageHeight) {
  if (!layout || !imageWidth || !imageHeight) {
    return null;
  }

  const containerRatio = layout.width / layout.height;
  const imageRatio = imageWidth / imageHeight;

  if (imageRatio >= containerRatio) {
    const displayWidth = layout.width;
    const displayHeight = displayWidth / imageRatio;
    return {
      displayWidth,
      displayHeight,
      offsetX: 0,
      offsetY: (layout.height - displayHeight) / 2
    };
  }

  const displayHeight = layout.height;
  const displayWidth = displayHeight * imageRatio;
  return {
    displayWidth,
    displayHeight,
    offsetX: (layout.width - displayWidth) / 2,
    offsetY: 0
  };
}

export function getSourcePointFromContainedPoint(
  locationX,
  locationY,
  imageBounds,
  imageWidth,
  imageHeight
) {
  if (!imageBounds || !imageWidth || !imageHeight) {
    return null;
  }

  const relativeX = locationX - imageBounds.offsetX;
  const relativeY = locationY - imageBounds.offsetY;

  if (
    relativeX < 0 ||
    relativeY < 0 ||
    relativeX > imageBounds.displayWidth ||
    relativeY > imageBounds.displayHeight
  ) {
    return null;
  }

  return {
    x: clamp(
      (relativeX / imageBounds.displayWidth) * imageWidth,
      0,
      imageWidth
    ),
    y: clamp(
      (relativeY / imageBounds.displayHeight) * imageHeight,
      0,
      imageHeight
    )
  };
}

export function getPrecisionOverlayGeometry({
  focusPoint,
  imageWidth,
  imageHeight,
  viewportWidth,
  viewportHeight,
  zoomScale
}) {
  if (
    !focusPoint ||
    !imageWidth ||
    !imageHeight ||
    !viewportWidth ||
    !viewportHeight ||
    !zoomScale
  ) {
    return null;
  }

  const zoomedWidth = imageWidth * zoomScale;
  const zoomedHeight = imageHeight * zoomScale;
  const minLeft = Math.min(0, viewportWidth - zoomedWidth);
  const maxLeft = Math.max(0, viewportWidth - zoomedWidth);
  const minTop = Math.min(0, viewportHeight - zoomedHeight);
  const maxTop = Math.max(0, viewportHeight - zoomedHeight);

  return {
    imageLeft: clamp(
      viewportWidth / 2 - focusPoint.x * zoomScale,
      minLeft,
      maxLeft
    ),
    imageTop: clamp(
      viewportHeight / 2 - focusPoint.y * zoomScale,
      minTop,
      maxTop
    ),
    imageHeight: zoomedHeight,
    imageWidth: zoomedWidth,
    zoomScale
  };
}

export function getSourcePointFromPrecisionTap(
  locationX,
  locationY,
  geometry,
  imageWidth,
  imageHeight
) {
  if (!geometry || !imageWidth || !imageHeight) {
    return null;
  }

  const zoomedWidth = geometry.imageWidth || imageWidth * geometry.zoomScale;
  const zoomedHeight = geometry.imageHeight || imageHeight * geometry.zoomScale;
  if (
    locationX < geometry.imageLeft ||
    locationY < geometry.imageTop ||
    locationX > geometry.imageLeft + zoomedWidth ||
    locationY > geometry.imageTop + zoomedHeight
  ) {
    return null;
  }

  return {
    x: clamp(
      (locationX - geometry.imageLeft) / geometry.zoomScale,
      0,
      imageWidth
    ),
    y: clamp(
      (locationY - geometry.imageTop) / geometry.zoomScale,
      0,
      imageHeight
    )
  };
}
