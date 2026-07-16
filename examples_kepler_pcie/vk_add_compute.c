// ponytail: minimal Vulkan compute vector add via NVK (Kepler GK104).
// gcc -O2 -o vk_add_compute vk_add_compute.c -lvulkan
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define N 64
#define CHECK(x) do { VkResult _r = (x); if (_r) { \
  fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

static uint32_t *load_spv(const char *path, size_t *words) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return NULL; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint32_t *buf = malloc(n);
  if (!buf || fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
  fclose(f);
  *words = (size_t)n / 4;
  return buf;
}

static uint32_t find_host_mem(VkPhysicalDevice pdev) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
      return i;
  }
  return UINT32_MAX;
}

int main(int argc, char **argv) {
  const char *spv = argc > 1 ? argv[1] : "add.spv";
  VkInstance inst;
  VkApplicationInfo app = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .apiVersion = VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app};
  CHECK(vkCreateInstance(&ici, NULL, &inst));

  uint32_t ndev = 0;
  CHECK(vkEnumeratePhysicalDevices(inst, &ndev, NULL));
  if (!ndev) { fprintf(stderr, "no GPU\n"); return 1; }
  VkPhysicalDevice *pdevs = calloc(ndev, sizeof(*pdevs));
  CHECK(vkEnumeratePhysicalDevices(inst, &ndev, pdevs));

  VkPhysicalDevice pdev = VK_NULL_HANDLE;
  for (uint32_t i = 0; i < ndev; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdevs[i], &props);
    printf("physdev[%u]: %s vendor=0x%x device=0x%x\n",
           i, props.deviceName, props.vendorID, props.deviceID);
    if (props.vendorID == 0x10de && props.deviceID == 0x1183)
      pdev = pdevs[i];
  }
  if (!pdev) pdev = pdevs[0];
  free(pdevs);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pdev, &props);
  printf("using: %s\n", props.deviceName);
  if (props.vendorID != 0x10de || props.deviceID != 0x1183)
    fprintf(stderr, "warning: not GTX 660 Ti (1183); got device=0x%x\n", props.deviceID);

  uint32_t qfc = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qfc, NULL);
  VkQueueFamilyProperties *qfp = calloc(qfc, sizeof(*qfp));
  vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qfc, qfp);
  uint32_t qf = UINT32_MAX;
  for (uint32_t i = 0; i < qfc; i++)
    if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
  free(qfp);
  if (qf == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }

  float prio = 1.f;
  VkDeviceQueueCreateInfo qci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio};
  VkDeviceCreateInfo dci = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
  VkDevice dev;
  CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, qf, 0, &queue);

  uint32_t host_mt = find_host_mem(pdev);
  if (host_mt == UINT32_MAX) { fprintf(stderr, "no host-visible memory\n"); return 1; }

  VkDeviceSize bytes = N * sizeof(float);
  VkBufferCreateInfo bci = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = bytes,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkBuffer ba, bb, bc;
  CHECK(vkCreateBuffer(dev, &bci, NULL, &ba));
  CHECK(vkCreateBuffer(dev, &bci, NULL, &bb));
  CHECK(vkCreateBuffer(dev, &bci, NULL, &bc));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, ba, &req);
  VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size, .memoryTypeIndex = host_mt};
  VkDeviceMemory ma, mb, mc;
  CHECK(vkAllocateMemory(dev, &mai, NULL, &ma));
  CHECK(vkAllocateMemory(dev, &mai, NULL, &mb));
  CHECK(vkAllocateMemory(dev, &mai, NULL, &mc));
  CHECK(vkBindBufferMemory(dev, ba, ma, 0));
  CHECK(vkBindBufferMemory(dev, bb, mb, 0));
  CHECK(vkBindBufferMemory(dev, bc, mc, 0));

  float *pa, *pb, *pc;
  CHECK(vkMapMemory(dev, ma, 0, bytes, 0, (void **)&pa));
  CHECK(vkMapMemory(dev, mb, 0, bytes, 0, (void **)&pb));
  CHECK(vkMapMemory(dev, mc, 0, bytes, 0, (void **)&pc));
  for (int i = 0; i < N; i++) {
    pa[i] = (float)i;
    pb[i] = (float)(i * 2);
    pc[i] = -1.f;
  }

  size_t nwords = 0;
  uint32_t *code = load_spv(spv, &nwords);
  if (!code) return 1;
  VkShaderModuleCreateInfo smci = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = nwords * 4, .pCode = code};
  VkShaderModule sm;
  CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
  free(code);

  VkDescriptorSetLayoutBinding binds[3];
  for (int i = 0; i < 3; i++)
    binds[i] = (VkDescriptorSetLayoutBinding){
      .binding = (uint32_t)i,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dslci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 3, .pBindings = binds};
  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
  VkPipelineLayoutCreateInfo plci = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1, .pSetLayouts = &dsl};
  VkPipelineLayout layout;
  CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
  VkComputePipelineCreateInfo cpci = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
    .layout = layout};
  VkPipeline pipe;
  CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

  VkDescriptorPoolSize dps = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3};
  VkDescriptorPoolCreateInfo dpci = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps};
  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
  VkDescriptorSet ds;
  CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo bis[3] = {
    {.buffer = ba, .range = bytes},
    {.buffer = bb, .range = bytes},
    {.buffer = bc, .range = bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++)
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = (uint32_t)i, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bis[i]};
  vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);

  VkCommandPoolCreateInfo cpoci = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = qf};
  VkCommandPool cmdpool;
  CHECK(vkCreateCommandPool(dev, &cpoci, NULL, &cmdpool));
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = cmdpool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1};
  VkCommandBuffer cmd;
  CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

  /* Host wrote A/B via coherent map → compute must see them; then host reads C. */
  VkMemoryBarrier host_to_shader = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
  VkMemoryBarrier shader_to_host = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_HOST_READ_BIT};

  VkCommandBufferBeginInfo cbbi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  CHECK(vkBeginCommandBuffer(cmd, &cbbi));
  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &host_to_shader, 0, NULL, 0, NULL);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, NULL);
  vkCmdDispatch(cmd, N, 1, 1);
  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
    0, 1, &shader_to_host, 0, NULL, 0, NULL);
  CHECK(vkEndCommandBuffer(cmd));

  VkFence fence;
  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  CHECK(vkCreateFence(dev, &fci, NULL, &fence));
  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1, .pCommandBuffers = &cmd};
  CHECK(vkQueueSubmit(queue, 1, &si, fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000));

  int ok = 1;
  for (int i = 0; i < N; i++)
    if (pc[i] != (float)(i * 3)) { ok = 0; break; }
  printf("first 5: %.0f %.0f %.0f %.0f %.0f\n", pc[0], pc[1], pc[2], pc[3], pc[4]);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
