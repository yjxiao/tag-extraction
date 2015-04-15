--[[
Dataset converter from csv to t7
Modified from code by Xiang Zhang
--]]

require("io")
require("os")
require("math")
require("paths")
require("torch")


-- Configuration table
config = {}
config.input = "train.csv"
config.output = "./"

-- Parse arguments
cmd = torch.CmdLine()
cmd:option("-input", config.input, "Input csv file")
cmd:option("-output", config.output, "Output t7b file")
params = cmd:parse(arg)
config.input = params.input
config.output = params.output
config.maxwords = 100
config.excludeCode = true

-- Check file existence
if not paths.filep(config.input) then
   error("Input file "..config.input.." does not exist.")
end
if paths.filep(config.output) then
   io.write("Output file "..config.output.." already exists. Do you want to continue? (y/[n]) ")
   local response = io.read("*line")
   if response:gsub("^%s*(.-)%s*$", "%1") ~= "y" then
      print("Exited because output file "..config.output.." already exists.")
      os.exit()
   end
end

-- truncate the string to maintain only the first k words
function kthwords(str, num)
   local k = 0
   local j = 0
   for i = 1, num do
      j, k = string.find(str, '%s+', k+1)
      if k == nil then
	 return str
      end
   end
   return str:sub(1, j-1)
end

-- Parser function for csv
-- Reference: http://lua-users.org/wiki/LuaCsv
function ParseCSVLine (line,sep) 
   local res = {}
   local pos = 1
   sep = sep or ','
   while true do 
      local c = string.sub(line,pos,pos)
      if (c == "") then break end
      if (c == '"') then
	 -- quoted value (ignore separator within)
	 local txt = ""
	 repeat
	    local startp,endp = string.find(line,'^%b""',pos)
	    txt = txt..string.sub(line,startp+1,endp-1)
	    pos = endp + 1
	    c = string.sub(line,pos,pos) 
	    if (c == '"') then txt = txt..'"' end 
	    -- check first char AFTER quoted string, if it is another
	    -- quoted string without separator, then append it
	    -- this is the way to "escape" the quote char in a quote. example:
	    --   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
	 until (c ~= '"')
	 table.insert(res,txt)
	 assert(c == sep or c == "")
	 pos = pos + 1
      else
	 -- no quotes used, just look for the first separator
	 local startp,endp = string.find(line,sep,pos)
	 if (startp) then 
	    table.insert(res,string.sub(line,pos,startp-1))
	    pos = endp + 1
	 else
	    -- no separator found -> use rest of string and terminate
	    table.insert(res,string.sub(line,pos))
	    break
	 end 
      end
   end
   return res
end

print("--- PASS 1: Checking file format and counting samples ---")
count = {}
n = 0
text = {}
label = {}
fd = io.open(config.input, 'r')
for line in fd:lines() do
   n = n + 1
   local content = ParseCSVLine(line)
   nitems = nitems or #content
   if nitems ~= 4 then
      error("Number of items not equal to 4")
   end
   if nitems ~= #content then
      error("Inconsistent number of items at line "..n)
   end

   local index = tonumber(content[1])
   if not index then
      goto continue
   end
   if index <= 0 then
      error("Index smaller than 1 at line "..n)
   end

   local class = content[4]:split(' ')
   for i = 1, #class do
      count[class[i]] = count[class[i]] or 0
      count[class[i]] = count[class[i]] + 1
   end
   label[index] = class

   if config.excludeCode then
      content[3] = content[3]:gsub("<code>.-</code>", ""):gsub("<.->", "")
   end
   local input_data = content[2]..' '..content[3]
   if config.maxwords then
      input_data = kthwords(input_data, config.maxwords)
   end
   text[index] = input_data

   if math.fmod(n, 10000) == 0 then
      io.write("\rProcessing line "..n)
      io.flush()
      collectgarbage()
   end
   ::continue::
end
fd:close()
collectgarbage()
print("\rNumber of lines processed: "..n)

torch.save(config.output..'input_text.t7', text)
text = nil
collectgarbage()
torch.save(config.output..'labels.t7', label)
print("\rProcessing done")
