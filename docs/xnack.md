# AMD XNACK: Retry-on-Fault and Unified Memory

This document explains the technical architecture of AMD's **XNACK** mechanism, its role in Heterogeneous Memory Management (HMM), and how to enable it on GFX9 (Vega/MI50) and newer hardware.

---

## 1. Overview: What is XNACK?

**XNACK** stands for **X**ecute-**N**o-**ACK**nowledge. It is a hardware feature in AMD GPUs that allows them to handle **GPU Page Faults** gracefully.

### 1.1 The Problem: Non-Retryable Faults
By default, most GPUs (including MI50 in its default state) are "non-retryable." If a GPU thread attempts to access a memory address that is not currently mapped in the GPU's Page Tables (a "page fault"), the hardware cannot pause the instruction. It simply generates an asynchronous interrupt and the driver kills the process (Segment Fault/Illegal Access).

### 1.2 The Solution: Retry-on-Fault
When XNACK is enabled, the GPU hardware gains the ability to **replay** a failed memory instruction.
1.  **Fault:** A GPU thread accesses a missing page.
2.  **Wait:** The thread is paused at the instruction level.
3.  **Handle:** The Linux Kernel (via HMM/KFD) catches the fault, fetches the page from Host RAM (or NVMe), and updates the GPU page tables.
4.  **Retry:** The GPU "retries" the exact same instruction, which now succeeds.

---

## 2. Theoretical Benefits

XNACK is the foundational technology for **System-wide SVM (Shared Virtual Memory)**.

*   **Lazy Loading:** Data is only moved to VRAM when a specific GPU core actually touches it.
*   **Zero-Copy (HMM):** GPUs can directly access arbitrary pointers returned by `malloc` or `mmap` without calling `hipHostRegister` or `hipMemcpy`.
*   **Over-subscription:** You can run models larger than physical VRAM (e.g., a 64GB model on a 32GB MI50) by faulting pages in and out of Host RAM automatically.
*   **NVMe-to-GPU Pipeline:** Files mapped via `mmap` can be processed by the GPU directly from the OS Page Cache.

---

## 3. The Three-Layer Enablement Stack

Enabling XNACK is a "full-stack" configuration. If any layer is missing, the system defaults to `xnack-` (disabled).

### Layer 1: BIOS (IOMMUv2)
XNACK relies on **IOMMUv2** features to communicate with the CPU's MMU.
*   **ATS (Address Translation Services):** Allows the GPU to ask the IOMMU for translations.
*   **PASID (Process Address Space ID):** Allows the GPU to identify which process a memory request belongs to.
*   **Action:** Enable `IOMMU`, `AMD-Vi`, and `AER` (Advanced Error Reporting) in BIOS.

### Layer 2: Kernel (`amdgpu.noretry`)
The `amdgpu` driver defaults to `noretry=1` (disabled) on many Vega-based cards to maximize raw throughput.
*   **Action:** Add `amdgpu.noretry=0` to your kernel boot parameters.
*   **Implementation:**
    ```bash
    # Edit /etc/default/grub
    GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.noretry=0"
    sudo update-grub && sudo reboot
    ```

### Layer 3: Runtime (`HSA_XNACK`)
Even with hardware support, the ROCm runtime must be told to initialize the GPU context in "retry" mode.
*   **Action:** `export HSA_XNACK=1`
*   **Verification:** Run `rocminfo | grep Name`. If you see `gfx906:sramecc+:xnack+`, it is successfully active.

---

## 4. Performance Trade-offs

Enabling XNACK is not "free." This is why it is often disabled by default in high-performance computing (HPC) environments.

1.  **Instruction Latency:** The hardware overhead of being able to replay instructions can reduce raw compute throughput by **5% to 15%**.
2.  **Synchronous Stalls:** A page fault on one thread can stall an entire Wavefront/Warp while the kernel handles the page migration.
3.  **PCIe Bottleneck:** Lazy loading over PCIe is significantly slower than local HBM2 access. If a kernel has low arithmetic intensity, it will spend 99% of its time waiting for the bus.

---

## 5. Application in Kernel-Fuser

In the `hip-explore` prototype, XNACK allows for two distinct modes of memory handling:

| Feature | XNACK Disabled (`xnack-`) | XNACK Enabled (`xnack+`) |
| :--- | :--- | :--- |
| **`hipMallocManaged`** | Works (Driver-driven migration) | Works (True Demand Paging) |
| **`mmap` Direct Access** | **CRASH** (Page Fault) | **SUCCESS** (Lazy load from Disk) |
| **`malloc` Direct Access** | **CRASH** | **SUCCESS** |
| **Workaround** | Must use `hipHostRegister` (Pins) | No registration needed |

---

## 6. Summary for MI50 (gfx906)

On the current system, the MI50s report `xnack-`. To achieve the "NVMe -> GPU" lazy loading requested in the prototypes without manual pinning, the **Layer 2 (Kernel)** fix must be applied. 

Without `amdgpu.noretry=0`, the hardware will continue to treat every page miss as a fatal error, regardless of the `HSA_XNACK=1` environment variable.
