/**
 * Regression: issue #291
 *
 * Plaud Ogg/Opus recordings can be rejected by OpenAI's diarization model even
 * when the container is valid, and the provider rejects any generated chunk
 * longer than 1,400 seconds. Riffado now normalizes diarization input to MP3
 * and splits long recordings into balanced chunks of at most 20 minutes.
 */
import type { OpenAI } from "openai";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { transcodeToMp3, transcodeToMp3Segments } = vi.hoisted(() => ({
    transcodeToMp3: vi.fn(),
    transcodeToMp3Segments: vi.fn(),
}));

vi.mock("@/lib/transcription/ffmpeg", () => ({
    transcodeToMp3,
    transcodeToMp3Segments,
}));

import { transcribeOpenAIDiarized } from "@/lib/transcription/openai-diarized-transcribe";

describe("issue #291 — OpenAI diarization audio preparation", () => {
    const audioBuffer = Buffer.from("plaud-ogg-opus");
    const create = vi.fn();
    const client = {
        audio: { transcriptions: { create } },
    } as unknown as OpenAI;

    beforeEach(() => {
        vi.clearAllMocks();
        transcodeToMp3.mockResolvedValue(Buffer.from("normalized-mp3"));
        transcodeToMp3Segments.mockResolvedValue([
            Buffer.from("chunk-1"),
            Buffer.from("chunk-2"),
        ]);
    });

    it("normalizes a short recording to MP3 before transcription", async () => {
        create.mockResolvedValue({
            segments: [{ speaker: "speaker_0", text: "Hello" }],
        });

        const result = await transcribeOpenAIDiarized({
            client,
            model: "gpt-4o-transcribe-diarize",
            audioBuffer,
            durationMs: 60_000,
            filename: "meeting.ogg",
            timeoutMs: 10_000,
        });

        expect(transcodeToMp3).toHaveBeenCalledWith(audioBuffer);
        expect(transcodeToMp3Segments).not.toHaveBeenCalled();
        expect(create).toHaveBeenCalledTimes(1);
        const params = create.mock.calls[0]?.[0];
        expect(params.file).toMatchObject({
            name: "meeting.mp3",
            type: "audio/mpeg",
        });
        expect(params.response_format).toBe("diarized_json");
        expect(params.chunking_strategy).toBe("auto");
        expect(result.text).toBe("speaker_0: Hello");
    });

    it("splits a 1,483 second recording into balanced sub-limit chunks", async () => {
        create
            .mockResolvedValueOnce({
                segments: [{ speaker: "speaker_0", text: "First" }],
            })
            .mockResolvedValueOnce({
                segments: [{ speaker: "speaker_0", text: "Second" }],
            });

        const result = await transcribeOpenAIDiarized({
            client,
            model: "gpt-4o-transcribe-diarize",
            audioBuffer,
            durationMs: 1_483_920,
            filename: "long-meeting.ogg",
            language: "en",
            timeoutMs: 10_000,
        });

        expect(transcodeToMp3Segments).toHaveBeenCalledOnce();
        expect(transcodeToMp3Segments).toHaveBeenCalledWith(
            audioBuffer,
            741.96,
        );
        expect(create).toHaveBeenCalledTimes(2);
        expect(create.mock.calls[0]?.[0].file.name).toBe(
            "long-meeting-part-1.mp3",
        );
        expect(create.mock.calls[1]?.[0].file.name).toBe(
            "long-meeting-part-2.mp3",
        );
        expect(create.mock.calls[0]?.[0].language).toBe("en");
        expect(result.text).toBe(
            "part_1_speaker_0: First\npart_2_speaker_0: Second",
        );
    });

    it("keeps every generated chunk at or below 20 minutes", async () => {
        transcodeToMp3Segments.mockResolvedValue([
            Buffer.from("chunk-1"),
            Buffer.from("chunk-2"),
            Buffer.from("chunk-3"),
            Buffer.from("chunk-4"),
        ]);
        create.mockResolvedValue({ segments: [] });

        await transcribeOpenAIDiarized({
            client,
            model: "gpt-4o-transcribe-diarize",
            audioBuffer,
            durationMs: 3_603_000,
            filename: "one-hour-recording.ogg",
            timeoutMs: 10_000,
        });

        expect(transcodeToMp3Segments).toHaveBeenCalledOnce();
        expect(transcodeToMp3Segments.mock.calls[0]?.[1]).toBeLessThanOrEqual(
            20 * 60,
        );
        expect(create).toHaveBeenCalledTimes(4);
    });
});
