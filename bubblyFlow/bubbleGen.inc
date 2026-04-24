#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

std::vector<dfloat> generateBubbleCenters(
    const int nBubbles,
    const dfloat rBubble,
    const dfloat xmin, const dfloat xmax,
    const dfloat ymin, const dfloat ymax,
    const dfloat zmin, const dfloat zmax,
    const dfloat wallBuffer,    // surface-to-wall distance
    const dfloat bubbleBuffer,  // surface-to-surface distance
    const unsigned seed = 12345)
{
  if (nBubbles < 0)
    throw std::runtime_error("nBubbles must be non-negative.");

  if (rBubble <= 0)
    throw std::runtime_error("Bubble radius must be positive.");

  if (xmax <= xmin || ymax <= ymin || zmax <= zmin)
    throw std::runtime_error("Invalid domain extents.");

  const dfloat xminC = xmin + rBubble + wallBuffer;
  const dfloat xmaxC = xmax - rBubble - wallBuffer;

  const dfloat yminC = ymin + rBubble + wallBuffer;
  const dfloat ymaxC = ymax - rBubble - wallBuffer;

  const dfloat zminC = zmin + rBubble + wallBuffer;
  const dfloat zmaxC = zmax - rBubble - wallBuffer;

  if (xminC > xmaxC || yminC > ymaxC || zminC > zmaxC)
    throw std::runtime_error("Domain too small for bubble radius and wall buffer.");

  std::mt19937 gen(seed);

  std::uniform_real_distribution<dfloat> distX(xminC, xmaxC);
  std::uniform_real_distribution<dfloat> distY(yminC, ymaxC);
  std::uniform_real_distribution<dfloat> distZ(zminC, zmaxC);

  std::vector<dfloat> coord;
  coord.reserve(3 * nBubbles);

  const dfloat minCenterDist = 2.0 * rBubble + bubbleBuffer;
  const dfloat minCenterDist2 = minCenterDist * minCenterDist;

  const int maxAttempts = 1000000;
  int attempts = 0;

  while ((int)(coord.size() / 3) < nBubbles && attempts < maxAttempts) {
    ++attempts;

    const dfloat x = distX(gen);
    const dfloat y = distY(gen);
    const dfloat z = distZ(gen);

    bool accept = true;

    for (int i = 0; i < (int)coord.size(); i += 3) {
      const dfloat dx = x - coord[i + 0];
      const dfloat dy = y - coord[i + 1];
      const dfloat dz = z - coord[i + 2];

      const dfloat dist2 = dx * dx + dy * dy + dz * dz;

      if (dist2 < minCenterDist2) {
        accept = false;
        break;
      }
    }

    if (accept) {
      coord.push_back(x);
      coord.push_back(y);
      coord.push_back(z);
    }
  }

  if ((int)(coord.size() / 3) != nBubbles)
    throw std::runtime_error("Could not place all bubbles. Reduce nBubbles, radius, or buffers.");

  return coord;
}
