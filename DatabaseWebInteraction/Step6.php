<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Volunteer Search</title>
</head>
<body>
<table>
<?php
$mysqli = new mysqli('localhost', '', '', '194_3430_project_team1');
if ($mysqli -> connect_errno)
{
    print "Error connecting to server";
    exit;
}
print <<<HTML_FORM
<form method ="get" action="$_SERVER[PHP_SELF]">
<p>Last Name: <input type="text" name='lname'><br>
<input type="submit" value="Submit">
</p>
</form>

HTML_FORM;

if ($_GET['lname'])
{
    $name = $_GET['lname'];
    $sql = "SELECT * FROM `volunteer` INNER JOIN person ON volunteer.person_id = person.person_id WHERE last_name = \"$name"";
    if (!$result = $mysqli->query($sql))
    {
        print "Error in query.";
        exit;
    }
	
	if (mysqli_num_rows($result) == 0)
	{
		print "<tr><th>No matching data found.</th></tr>";
	}
	else
	{
		print "<tr><th>Person_ID</th><th>First Name</th><th>Last Name</th><th>Preferred Name</th><th>Email Address</th><th>Previous Experience</th>
		<th>Preferrerd Weekly Hours</th> <th>Occupation</th> <th>Birthdate</th></tr>";
		while($d = $result->fetch_assoc())
		{
			print "Got inside the while loop";
			$person_id = $d['person_id']; $prevExperience = $d['previous_experience']; $preWeeky = $d['preferred_weekly_hours']; $occupation = $d['occupation']; $fname = $d['first_name'];
			$lname = $d['last_name']; $preName = $d['preferred_name']; $email = $d['email_address']; $birthdate = $d['birthdate'];
		
			print "<tr><td>$person_id</td><td>$fname</td><td>$lname</td><td>$preName</td><td>$email</td><td>$prevExperience</td><td>$preWeeky</td>
			<td>$occupation</td><td>$birthdate</td></tr>\n";
		}
	}
	print "</table>\n";
    $result ->free();
}   
$mysqli->close();
print <<<HTML_END
</body>
</html>
HTML_END;

?>
