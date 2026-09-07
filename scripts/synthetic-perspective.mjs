/** Bounded whole-page homography; coordinates are image-edge coordinates. */
export function perspectiveRecipe(width, height, index) {
  if (index >= 100000 || index % 5 !== 4) return null;
  return [
    0.94,
    0,
    width * 0.02,
    0,
    0.94,
    height * 0.02,
    (index % 2 ? 0.035 : -0.025) / width,
    0.025 / height,
    1,
  ];
}

export function projectPoint(matrix, [x, y]) {
  const z = matrix[6] * x + matrix[7] * y + matrix[8];
  if (!Number.isFinite(z) || z <= 0)
    throw new Error("invalid projective denominator");
  return [
    (matrix[0] * x + matrix[1] * y + matrix[2]) / z,
    (matrix[3] * x + matrix[4] * y + matrix[5]) / z,
  ];
}

export function validatePerspective(matrix, width, height, index) {
  if (
    JSON.stringify(matrix) !==
    JSON.stringify(perspectiveRecipe(width, height, index))
  )
    throw new Error("unapproved page perspective");
}

export function perspectiveCss(m) {
  if (!m) return "none";
  return `matrix3d(${[m[0], m[3], 0, m[6], m[1], m[4], 0, m[7], 0, 0, 1, 0, m[2], m[5], 0, m[8]].join(",")})`;
}
