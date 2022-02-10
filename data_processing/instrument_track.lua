-- instrument_track.lua v0.2 by Christian Fillion (cfillion)

MAX_CHANNEL_COUNT = 32 -- set this to 64 if you need to use two MIDI busses
MAX_MIDI_BUS = MAX_CHANNEL_COUNT / (16 * 2)

function GetTrackMidiReceives(track)
  local list, index, recvIndex, recvCount

  list, index, recvIndex = {}, 0, 0
  recvCount = reaper.GetTrackNumSends(track, -1)

  while recvIndex < recvCount do
    local dstBus, dstChan

    dstBus = reaper.BR_GetSetTrackSendInfo(
      track, -1, recvIndex, "I_MIDI_DSTBUS", false, 0)
    dstChan = reaper.BR_GetSetTrackSendInfo(
      track, -1, recvIndex, "I_MIDI_DSTCHAN", false, 0)

    if dstBus + dstChan > -1 then
      if dstBus == 0 then dstBus = 1 end
      list[index] = {BUS=dstBus, CHAN=dstChan}
      index = index + 1
    end

    recvIndex = recvIndex + 1
  end

  return list
end

function GetUnusedMidiChannel(sends)
  local index, busses

  busses = {}

  index = 1
  while index <= MAX_MIDI_BUS do
    busses[index] = 0
    index = index + 1
  end

  for _,send in pairs(sends) do
    local bus = math.floor(send["BUS"])
    local chan = math.floor(send["CHAN"])

    if busses[bus] ~= nil and busses[bus] < chan then
      busses[bus] = chan
    end
  end

  for i,chan in pairs(busses) do
    if chan <= 15 then
      return i,chan+1
    end
  end

  return -1, -1
end

function GetUnusedSlot()
  local trackId, smplId, sampler, sends, bus, chan

  bus, chan = -1, -1

  while chan == -1 do
    trackId, smplId, sampler = GetLastSampler()

    if trackId == -1 then
      smplId = 1
      sampler = InsertSamplerAt(0, 1)
    end

    sends = GetTrackMidiReceives(sampler)
    bus, chan = GetUnusedMidiChannel(sends)

    if chan == -1 then
      if smplId == 1 then
        -- append the omitted sampler ID to the first sampler track
        reaper.GetSetMediaTrackInfo_String(sampler, "P_NAME", "Sampler 1", true)
      end

      InsertSamplerAt(trackId+1, smplId+1)
    end
  end

  return sampler, smplId, bus, chan
end

function GetLastSampler()
  local index, trackCount = 0, reaper.GetNumTracks()
  local trackId, lastId, sampler = -1, 0, nil

  while index < trackCount do
    local track = reaper.GetTrack(0, index)
    local _, name = reaper.GetSetMediaTrackInfo_String(
      track, "P_NAME", "", false)

    local match = string.find(name, "^Sampler%s?")
    if match ~= nil then
      local id = tonumber(string.sub(name, 9))
      if id == nil then id = 1 end

      if lastId < id then
        trackId = index
        lastId = id
        sampler = track
      end
    end

    index = index + 1
  end

  return trackId, lastId, sampler
end

function InsertSamplerAt(index, id)
  reaper.InsertTrackAtIndex(index, false)
  track = reaper.GetTrack(0, index)

  local name = "Sampler"
  if id > 1 then name = string.format("%s %d", name, id) end

  reaper.GetSetMediaTrackInfo_String(track, "P_NAME", name, true)
  reaper.SetMediaTrackInfo_Value(track, "B_MAINSEND", 0)
  reaper.SetMediaTrackInfo_Value(track, "B_SHOWINTCP", 0)
  reaper.SetMediaTrackInfo_Value(track, "I_NCHAN", MAX_CHANNEL_COUNT)

  -- set recording mode to multichannel output
  reaper.SetMediaTrackInfo_Value(track, "I_RECMODE", 10)

  return track
end

function MidiToAudioChannel(bus, chan)
  if bus == 1 then bus = 0 end
  return (bus * 16) + (chan - 1) * 2
end

function GetInsertionPoint()
  local selectionSize = reaper.CountSelectedTracks(0)
  if selectionSize == 0 then
    return reaper.GetNumTracks()
  end

  track = reaper.GetSelectedTrack(0, selectionSize - 1)
  return reaper.GetMediaTrackInfo_Value(track, "IP_TRACKNUMBER")
end

reaper.PreventUIRefresh(1)
reaper.Undo_BeginBlock()

local sampler, smplId, bus, chan = GetUnusedSlot()
local audioChan = MidiToAudioChannel(bus, chan)
local insertPos = GetInsertionPoint()

-- create AUDIO track
reaper.InsertTrackAtIndex(insertPos, true)
audioTrack = reaper.GetTrack(0, insertPos)
reaper.SetMediaTrackInfo_Value(audioTrack, "I_FOLDERDEPTH", 1)
reaper.SetMediaTrackInfo_Value(audioTrack, "I_HEIGHTOVERRIDE", 1)
reaper.SetMediaTrackInfo_Value(audioTrack, "I_RECMODE", 1)
reaper.GetSetMediaTrackInfo_String(audioTrack, "P_NAME",
  string.format("#%d %d/%d", smplId, audioChan+1, audioChan+2), true)

reaper.SNM_AddReceive(sampler, audioTrack, -1)
reaper.BR_GetSetTrackSendInfo(
  audioTrack, -1, 0, "I_SRCCHAN", true, audioChan)
reaper.BR_GetSetTrackSendInfo(
  audioTrack, -1, 0, "I_DSTCHAN", true, 0)
reaper.BR_GetSetTrackSendInfo(
  audioTrack, -1, 0, "I_MIDI_SRCCHAN", true, -1)

-- create MIDI track
reaper.InsertTrackAtIndex(insertPos+1, true)
midiTrack = reaper.GetTrack(0, insertPos+1)
reaper.SetMediaTrackInfo_Value(midiTrack, "B_SHOWINMIXER", 0)
reaper.SetMediaTrackInfo_Value(midiTrack, "I_FOLDERDEPTH", -1)
reaper.SetMediaTrackInfo_Value(midiTrack, "I_RECMON", 1)
reaper.GetSetMediaTrackInfo_String(midiTrack, "P_NAME",
  string.format("-> #%d B:%d C:%d", smplId, bus, chan), true)
reaper.SNM_AddReceive(midiTrack, sampler, 0)
reaper.BR_GetSetTrackSendInfo(
  midiTrack, 0, 0, "I_SRCCHAN", true, -1)
reaper.BR_GetSetTrackSendInfo(
  midiTrack, 0, 0, "I_MIDI_SRCBUS", true, 0)
reaper.BR_GetSetTrackSendInfo(
  midiTrack, 0, 0, "I_MIDI_SRCCHAN", true, 0)
reaper.BR_GetSetTrackSendInfo(
  midiTrack, 0, 0, "I_MIDI_DSTBUS", true, bus)
reaper.BR_GetSetTrackSendInfo(
  midiTrack, 0, 0, "I_MIDI_DSTCHAN", true, chan)

reaper.Undo_EndBlock("Create Instrument Track", 1)

reaper.PreventUIRefresh(-1)
reaper.TrackList_AdjustWindows(false)
