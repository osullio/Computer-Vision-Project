# Abandoned & Removed Object Detection

Computer Vision assignment (Year 4, TCD). An OpenCV/C++ program that detects objects that are
**abandoned** in or **removed** from a scene in surveillance video, and evaluates the detections
against hand-labelled ground truth.

## How it works

For each video, `processVideo` in `MyApplication.cpp`:

1. **Background subtraction:** a Gaussian Mixture Model (`BackgroundSubtractorMOG2`, shadows off)
   produces a foreground mask.
2. **Motion mask:** temporal differencing against a frame 5 frames back separates moving
   foreground from static foreground.
3. **Persistence:** pixels that stay foreground but not moving build up a persistence count. A region
   that stays still for about 6 seconds is a candidate stationary object.
4. **Filtering & tracking:** candidate contours are filtered by area (150–3000 px) and aspect ratio,
   tracked across frames, merged when they overlap, and confirmed after 15 consistent frames.
5. **Classification:** edges in the current frame are compared with an early-history baseline of the
   empty scene. Edges present now but not in the baseline indicate an **abandoned** object. Edges
   present in the baseline but missing now indicate a **removed** object.
6. **Evaluation:** detections are matched to ground truth (IoU ≥ 0.5, within ±120 frames).
   The program prints per-video and overall precision, recall and F1, along with detection and
   classification accuracy.

The program also displays the frame, GMM mask, motion mask and suspected stationary objects while it runs.

## Files

| File | Description |
|---|---|
| `MyApplication.cpp` | Main/final version of the solution |
| `testing.cpp` | Working copy used for experimentation (near-identical to `MyApplication.cpp`) |
| `Save.cpp` | Earlier saved version of the solution |
| `.vscode/` | VS Code build configuration (MSVC + OpenCV) |

## Building

Requirements: Windows, MSVC (Visual Studio 2022 Build Tools, x64) and OpenCV 4.12.

The included VS Code task (`.vscode/tasks.json`) expects OpenCV at `C:\opencv\build`. Update the paths there
and in `.vscode/c_cpp_properties.json` if your install is somewhere else. To build from a
*x64 Native Tools Command Prompt*:

```
cl /nologo /EHsc /Zi /Fe:output\MyApplication.exe MyApplication.cpp ^
   /I C:\opencv\build\include ^
   /link /MACHINE:X64 /LIBPATH:C:\opencv\build\x64\vc16\lib opencv_world4120.lib
```

`opencv_world4120.dll` needs to be on your `PATH` (or next to the executable) at runtime.

## Data

The input videos (`Video01.avi` – `Video10.avi`) were provided as part of the assignment and are
**not included** in this repository. To run the program, put them in the project root and run the
executable from there.

Note: Video 10 is excluded from processing because it caused the program to crash.
